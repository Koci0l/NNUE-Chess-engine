// src/search.cpp

#include "search.h"
#include "tt.h"
#include "history.h"
#include "movepick.h"
#include "see.h"
#include "zobrist.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

// ============================================================================
// LMR Table
// ============================================================================

static int lmr_reductions[64][64];

void initLMR() {
    for (int depth = 1; depth < 64; ++depth) {
        for (int move_num = 1; move_num < 64; ++move_num) {
            double reduction = 0.75 + std::log(depth) * std::log(move_num) / 2.25;
            lmr_reductions[depth][move_num] = static_cast<int>(reduction);
        }
    }
}

// ============================================================================
// Constants and Tunables
// ============================================================================

// Singular Extensions
constexpr int SE_MIN_DEPTH = 5;
constexpr int SE_DEPTH_TOL = 3;
constexpr int SE_MARGIN_PER_DEPTH = 2;
constexpr int SE_DOUBLE_BIAS = 55;
constexpr int SE_TRIPLE_BIAS = 120;

// ProbCut
constexpr int PROBCUT_MIN_DEPTH = 5;
constexpr int PROBCUT_BETA_MARGIN = 150;
constexpr int PROBCUT_DEPTH_SUBTRACTOR = 4;
constexpr int PROBCUT_IMPROVING_MARGIN = 30;
constexpr int PROBCUT_SEE_THRESHOLD = 100;
constexpr int PROBCUT_MAX_MOVES = 3;

// Small ProbCut
constexpr int SPROBCUT_BETA_MARGIN = 350;
constexpr int SPROBCUT_TT_DEPTH_SUBTRACTOR = 4;

// No-alloc move ordering buffers
constexpr int MAX_MOVES_NOALLOC = 256;     // legal moves <= 218, keep headroom
constexpr int MAX_QUIETS_TRACKED = 64;

// ============================================================================
// Helper Functions
// ============================================================================

inline int scaleNNUE(int raw_score) {
    return raw_score / 2;
}

bool isDrawByRepetition(const chess::Board& board) {
    return board.isRepetition(1);
}

bool isDrawByFiftyMove(const chess::Board& board) {
    return board.halfMoveClock() >= 100;
}

int getDrawScore(int /*ply_from_root*/) {
    return CONTEMPT;
}

// ============================================================================
// No-allocation move scoring + selection (replaces std::vector<ScoredMove> in search)
// ============================================================================

static inline void pickNextMoveNoAlloc(ScoredMove* moves, int count, int idx) {
    int best = idx;
    int bestScore = moves[idx].score;
    for (int j = idx + 1; j < count; ++j) {
        if (moves[j].score > bestScore) {
            best = j;
            bestScore = moves[j].score;
        }
    }
    if (best != idx) std::swap(moves[idx], moves[best]);
}

static inline int scoreMoveNoAlloc(const chess::Board& board,
                                   const chess::Move& move,
                                   chess::Move tt_move,
                                   chess::Move counter_move,
                                   chess::Color side_to_move,
                                   int ply_from_root) {
    // Big constants to force ordering tiers.
    constexpr int TT_BONUS      = 3'000'000;
    constexpr int CAPTURE_BONUS = 2'000'000;
    constexpr int KILLER_BONUS  = 1'500'000;
    constexpr int COUNTER_BONUS = 1'250'000;
    constexpr int PROMO_BONUS   = 600'000;

    if (move == tt_move) return TT_BONUS;

    const auto mt = move.typeOf();
    const bool is_ep = (mt == chess::Move::ENPASSANT);
    const bool is_promo = (mt == chess::Move::PROMOTION);

    chess::Piece captured = chess::Piece::NONE;
    if (is_ep) {
        // captured pawn is not on move.to()
        chess::Square capSq(move.to().file(), move.from().rank());
        captured = board.at(capSq);
    } else {
        captured = board.at(move.to());
    }
    const bool is_capture = (captured != chess::Piece::NONE);

    // Noisy
    if (is_capture || is_promo) {
        int victim = 0;
        if (is_capture) victim = pieceValue(captured.type());

        chess::Piece attackerP = board.at(move.from());
        int attacker = pieceValue(attackerP.type());

        // MVV-LVA-ish, stable and cheap.
        int score = CAPTURE_BONUS + victim * 10 - attacker;

        if (is_promo) {
            // Promotions should be searched early even if non-capture.
            score += PROMO_BONUS + pieceValue(move.promotionType());
        }

        return score;
    }

    // Quiet tiers
    if (g_killerMoves.is_killer(ply_from_root, move)) return KILLER_BONUS;
    if (move == counter_move) return COUNTER_BONUS;

    // History (can be negative)
    return g_butterflyHistory.get(side_to_move, move.from(), move.to());
}

static inline void scoreMovesNoAlloc(const chess::Movelist& moves,
                                     const chess::Board& board,
                                     chess::Move tt_move,
                                     chess::Move counter_move,
                                     chess::Color side_to_move,
                                     int ply_from_root,
                                     ScoredMove* out,
                                     int& outCount) {
    outCount = 0;
    for (const auto& mv : moves) {
        if (outCount >= MAX_MOVES_NOALLOC) break;
        out[outCount].move = mv;
        out[outCount].score = scoreMoveNoAlloc(board, mv, tt_move, counter_move, side_to_move, ply_from_root);
        ++outCount;
    }
}

// ============================================================================
// Accumulator Update
// ============================================================================

void updateAccumulatorForMove(AccumulatorStack& accStack, chess::Board& board,
                               const chess::Move& move) {
    auto moveType = move.typeOf();

    if (moveType == chess::Move::NORMAL) {
        chess::Piece piece = board.at(move.from());
        chess::Piece captured = board.at(move.to());
        if (captured != chess::Piece::NONE) {
            accStack.current().remove_piece(captured, move.to());
        }
        accStack.current().move_piece(piece, move.from(), move.to());
    } else if (moveType == chess::Move::PROMOTION) {
        chess::Piece pawn = board.at(move.from());
        chess::Piece captured = board.at(move.to());
        chess::Piece promotedPiece = chess::Piece(move.promotionType(), pawn.color());
        if (captured != chess::Piece::NONE) {
            accStack.current().remove_piece(captured, move.to());
        }
        accStack.current().remove_piece(pawn, move.from());
        accStack.current().add_piece(promotedPiece, move.to());
    } else if (moveType == chess::Move::ENPASSANT) {
        chess::Piece pawn = board.at(move.from());
        chess::Square capturedPawnSq = chess::Square(move.to().file(), move.from().rank());
        chess::Piece capturedPawn = board.at(capturedPawnSq);
        accStack.current().remove_piece(capturedPawn, capturedPawnSq);
        accStack.current().move_piece(pawn, move.from(), move.to());
    } else if (moveType == chess::Move::CASTLING) {
        chess::Piece pieces_before[64];
        for (int i = 0; i < 64; i++)
            pieces_before[i] = board.at(chess::Square(i));

        board.makeMove(move);

        for (int i = 0; i < 64; i++) {
            chess::Square sq(i);
            chess::Piece before = pieces_before[i];
            chess::Piece after = board.at(sq);
            if (before != after) {
                if (before != chess::Piece::NONE)
                    accStack.current().remove_piece(before, sq);
                if (after != chess::Piece::NONE)
                    accStack.current().add_piece(after, sq);
            }
        }
        board.unmakeMove(move);
    }
}

// ============================================================================
// Singular Extension
// ============================================================================

struct SEResult {
    int ext = 0;
    bool multicut = false;
    int mcScore = 0;
};

SEResult probeSingularExtension(chess::Board& board, int depth, int beta, int ply_from_root,
                                 ThreadInfo& thread, const TimeManager* tm, SearchStats& stats,
                                 chess::Move tt_move, uint64_t hash,
                                 bool is_pv_node, bool is_quiet_move, SearchStack* ss) {
    SEResult out;
    if (depth < SE_MIN_DEPTH || tt_move == chess::Move()) return out;

    TTEntry te;
    if (!peekTT(hash, te)) return out;
    if (te.best_move != tt_move) return out;
    if (te.flag == TT_UPPER) return out;
    if (te.depth < depth - SE_DEPTH_TOL) return out;

    int tt_score = te.score;
    if (tt_score >= MATE_SCORE - 100) tt_score -= ply_from_root;
    else if (tt_score <= -MATE_SCORE + 100) tt_score += ply_from_root;
    if (std::abs(tt_score) >= MATE_SCORE - 100) return out;

    int singular_beta = std::max(-MATE_SCORE + 1, tt_score - depth * SE_MARGIN_PER_DEPTH);
    int seDepth = (depth - 1) / 2;
    if (seDepth <= 0) return out;

    int val = alphaBeta(board, seDepth, singular_beta - 1, singular_beta,
                        ply_from_root, thread, tm, stats, false, chess::Move(), ss, tt_move);

    if (val < singular_beta) {
        int ext = 1;
        if (!is_pv_node && val + SE_DOUBLE_BIAS < singular_beta) ext += 1;
        if (!is_pv_node && is_quiet_move && val + SE_TRIPLE_BIAS < singular_beta) ext += 1;
        out.ext = ext;
        return out;
    }
    if (singular_beta >= beta) {
        out.multicut = true;
        out.mcScore = singular_beta;
        return out;
    }
    if (tt_score >= beta) out.ext = -1;
    return out;
}

// ============================================================================
// PV Extraction
// ============================================================================

std::vector<chess::Move> extractPV(chess::Board board, int max_depth) {
    std::vector<chess::Move> pv;

    for (int i = 0; i < max_depth; ++i) {
        uint64_t hash = getZobristHash(board);
        size_t index = hash & TT_MASK;
        const TTEntry& entry = tt[index];
        if (entry.key != hash || entry.best_move == chess::Move()) break;

        chess::Move move = entry.best_move;
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);

        bool is_legal = false;
        for (const auto& m : legal_moves) {
            if (m == move) {
                is_legal = true;
                break;
            }
        }
        if (!is_legal) break;

        pv.push_back(move);
        board.makeMove(move);

        chess::Movelist next_moves;
        chess::movegen::legalmoves(next_moves, board);
        if (next_moves.empty()) break;
    }
    return pv;
}

// ============================================================================
// Quiescence Search
// ============================================================================

int quiescence(chess::Board& board, int alpha, int beta,
               ThreadInfo& thread, int ply_from_root, SearchStats& stats) {
    stats.nodes++;

    bool in_check = board.inCheck();

    int stand_pat = 0;
    if (!in_check) {
        stand_pat = scaleNNUE(g_nnue.evaluate(board, thread));
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;
    }

    chess::Movelist all_moves;
    chess::movegen::legalmoves(all_moves, board);

    // No-allocation scored buffer
    ScoredMove scored_moves[MAX_MOVES_NOALLOC];
    int scored_count = 0;

    if (in_check) {
        if (all_moves.empty()) {
            return -MATE_SCORE + ply_from_root;
        }
        scoreMovesNoAlloc(all_moves, board, chess::Move(), chess::Move(), board.sideToMove(),
                          ply_from_root, scored_moves, scored_count);
    } else {
        chess::Movelist tactical_moves;
        for (const auto& move : all_moves) {
            bool is_tactical = board.at(move.to()) != chess::Piece::NONE ||
                               move.typeOf() == chess::Move::PROMOTION ||
                               move.typeOf() == chess::Move::ENPASSANT;
            if (is_tactical)
                tactical_moves.add(move);
        }
        scoreMovesNoAlloc(tactical_moves, board, chess::Move(), chess::Move(), board.sideToMove(),
                          ply_from_root, scored_moves, scored_count);
    }

    for (int i = 0; i < scored_count; ++i) {
        pickNextMoveNoAlloc(scored_moves, scored_count, i);
        const auto& move = scored_moves[i].move;

        bool is_tactical = board.at(move.to()) != chess::Piece::NONE ||
                           move.typeOf() == chess::Move::PROMOTION ||
                           move.typeOf() == chess::Move::ENPASSANT;

        // SEE pruning for non-check positions
        if (!in_check && is_tactical && !chess::see::see_ge(board, move, 0)) {
            continue;
        }

        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);

        int eval = -quiescence(board, -beta, -alpha, thread, ply_from_root + 1, stats);

        board.unmakeMove(move);
        thread.accumulatorStack.pop();

        if (eval >= beta) return beta;
        if (eval > alpha) alpha = eval;
    }

    return alpha;
}

// ============================================================================
// Alpha-Beta Search
// ============================================================================

int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int ply_from_root,
              ThreadInfo& thread, const TimeManager* tm, SearchStats& stats, bool allow_null,
              chess::Move previous_move, SearchStack* ss, chess::Move excluded_move) {
    stats.nodes++;

    // Time check
    if (tm && tm->should_stop()) return alpha;

    // Draw detection
    if (ply_from_root > 0) {
        if (isDrawByRepetition(board)) return getDrawScore(ply_from_root);
        if (isDrawByFiftyMove(board)) return getDrawScore(ply_from_root);
    }

    uint64_t hash = getZobristHash(board);
    int tt_score = 0;
    chess::Move tt_move;
    bool tt_pv = false;
    int tt_depth = 0;
    TTFlag tt_flag = TT_EXACT;

    bool in_singular_search = (excluded_move != chess::Move());

    // TT Probe
    if (!in_singular_search) {
        bool tt_hit = probeTT(hash, depth, alpha, beta, tt_score, tt_move, ply_from_root, tt_pv);
        if (tt_hit) {
            TTEntry te;
            if (peekTT(hash, te)) {
                tt_depth = te.depth;
                tt_flag = te.flag;
            }
            return tt_score;
        }
    } else {
        tt_move = chess::Move();
    }

    bool in_check = board.inCheck();
    bool is_pv_node = (beta - alpha) > 1;

    // Internal Iterative Reduction (IIR)
    if (!in_singular_search && depth >= 4 && tt_move == chess::Move() && !in_check) {
        depth--;
    }

    int static_eval = 0;
    bool improving = false;

    if (!in_check) {
        static_eval = scaleNNUE(g_nnue.evaluate(board, thread));
        ss[ply_from_root].static_eval = static_eval;

        // Calculate improving
        if (ply_from_root >= 2) {
            improving = ss[ply_from_root].static_eval > ss[ply_from_root - 2].static_eval;
        }
    }

    int extension = 0;
    if (in_check) {
        extension = 1;
    }

    // ========================================================================
    // Reverse Futility Pruning (RFP)
    // ========================================================================
    if (!is_pv_node &&
        !in_check &&
        depth <= 7 &&
        depth >= 1 &&
        std::abs(beta) < MATE_SCORE - 100) {

        int rfp_margin = 85 * depth;
        if (static_eval - rfp_margin >= beta) {
            return static_eval - rfp_margin;
        }
    }

    // ========================================================================
    // Null Move Pruning (NMP)
    // ========================================================================
    if (allow_null &&
        !in_check &&
        !is_pv_node &&
        depth >= 3 &&
        hasNonPawnMaterial(board) &&
        static_eval >= beta) {  // Only try NMP if static eval looks promising

        // Depth-dependent reduction: R = 3 + depth/3
        int R = 3 + depth / 3;
        R = std::min(R, depth - 1);  // Don't reduce below depth 1

        AccumulatorPair saved_acc = thread.accumulatorStack.current();
        board.makeNullMove();
        thread.accumulatorStack.push();
        thread.accumulatorStack.current() = saved_acc;

        ss[ply_from_root + 1].static_eval = -ss[ply_from_root].static_eval;

        int null_score = -alphaBeta(board, depth - R - 1, -beta, -beta + 1,
                                    ply_from_root + 1, thread, tm, stats, false,
                                    chess::Move(), ss);

        thread.accumulatorStack.pop();
        board.unmakeNullMove();

        if (null_score >= beta) {
            if (null_score >= MATE_SCORE - 100) return beta;
            return null_score;
        }
    }

    // ========================================================================
    // ProbCut
    // ========================================================================
    int probcut_beta = beta + PROBCUT_BETA_MARGIN - PROBCUT_IMPROVING_MARGIN * improving;

    if (!in_singular_search &&
        !is_pv_node &&
        !in_check &&
        depth >= PROBCUT_MIN_DEPTH &&
        std::abs(beta) < MATE_SCORE - 200 &&
        hasNonPawnMaterial(board)) {

        // TT gating - only run if no deep TT hit or TT suggests cutoff likely
        bool tt_gate = true;
        if (tt_move != chess::Move() && tt_depth >= depth - 3 && tt_score < probcut_beta) {
            tt_gate = false;
        }

        if (tt_gate) {
            int probcut_depth = depth - PROBCUT_DEPTH_SUBTRACTOR;

            // Generate only captures and promotions
            chess::Movelist all_moves;
            chess::movegen::legalmoves(all_moves, board);

            struct ProbCutMove {
                chess::Move mv;
                int score;
            };
            std::vector<ProbCutMove> probcut_moves;

            for (const auto& mv : all_moves) {
                bool is_capture = board.at(mv.to()) != chess::Piece::NONE || mv.typeOf() == chess::Move::ENPASSANT;
                bool is_promotion = mv.typeOf() == chess::Move::PROMOTION;

                if (!is_capture && !is_promotion) continue;

                // SEE pruning
                if (!chess::see::see_ge(board, mv, PROBCUT_SEE_THRESHOLD)) continue;

                int victim = 0;
                if (mv.typeOf() == chess::Move::ENPASSANT) {
                    victim = pieceValue(chess::PieceType::PAWN);
                } else if (board.at(mv.to()) != chess::Piece::NONE) {
                    victim = pieceValue(board.at(mv.to()).type());
                }

                int attacker = pieceValue(board.at(mv.from()).type());
                int move_score = victim * 10 - attacker + (is_promotion ? 500 : 0);

                probcut_moves.push_back({mv, move_score});
            }

            // Sort by score (best captures first)
            std::sort(probcut_moves.begin(), probcut_moves.end(),
                      [](const ProbCutMove& a, const ProbCutMove& b) {
                          return a.score > b.score;
                      });

            // Try best captures
            for (size_t i = 0; i < std::min(probcut_moves.size(), size_t(PROBCUT_MAX_MOVES)); ++i) {
                const auto& move = probcut_moves[i].mv;

                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);

                // First try qsearch to quickly see if this is promising
                int probcut_value = -quiescence(board, -probcut_beta, -probcut_beta + 1,
                                               thread, ply_from_root + 1, stats);

                // If qsearch passes, do reduced depth search
                if (probcut_value >= probcut_beta) {
                    probcut_value = -alphaBeta(board, probcut_depth - 1, -probcut_beta, -probcut_beta + 1,
                                               ply_from_root + 1, thread, tm, stats, false, move, ss);
                }

                board.unmakeMove(move);
                thread.accumulatorStack.pop();

                if (probcut_value >= probcut_beta) {
                    storeTT(hash, probcut_depth, probcut_value, move, TT_LOWER, ply_from_root, tt_pv);
                    return probcut_value;
                }

                if (tm && tm->should_stop()) break;
            }
        }
    }

    // ========================================================================
    // Small ProbCut - exploits deep TT entries
    // ========================================================================
    int small_probcut_beta = beta + SPROBCUT_BETA_MARGIN;

    if (!in_singular_search &&
        !is_pv_node &&
        tt_move != chess::Move() &&
        tt_flag == TT_LOWER &&
        tt_depth >= depth - SPROBCUT_TT_DEPTH_SUBTRACTOR &&
        tt_score >= small_probcut_beta &&
        std::abs(tt_score) < MATE_SCORE - 200 &&
        std::abs(beta) < MATE_SCORE - 200) {

        return small_probcut_beta;
    }

    // ========================================================================
    // Drop to Quiescence
    // ========================================================================
    if (depth + extension <= 0) {
        return quiescence(board, alpha, beta, thread, ply_from_root, stats);
    }

    // ========================================================================
    // Move Generation
    // ========================================================================
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        if (in_check) return -MATE_SCORE + ply_from_root;
        return getDrawScore(ply_from_root);
    }

    chess::Move counter_move = g_counterMoves.get(previous_move);
    chess::Color side_to_move = board.sideToMove();

    // No-allocation scored buffer (replaces std::vector<ScoredMove>)
    ScoredMove scored_moves[MAX_MOVES_NOALLOC];
    int scored_count = 0;
    scoreMovesNoAlloc(moves, board, tt_move, counter_move, side_to_move, ply_from_root,
                      scored_moves, scored_count);

    chess::Move best_move;
    int best_score = -MATE_SCORE;
    int original_alpha = alpha;

    chess::Move quiets_searched[MAX_QUIETS_TRACKED];
    int quiets_count = 0;

    int move_count = 0;

    // ========================================================================
    // Move Loop
    // ========================================================================
    for (int i = 0; i < scored_count; ++i) {
        pickNextMoveNoAlloc(scored_moves, scored_count, i);
        const auto& move = scored_moves[i].move;

        // Skip excluded move (for singular search)
        if (move == excluded_move) continue;

        bool is_quiet = isQuietMove(board, move);
        bool is_noisy = !is_quiet;

        // ====================================================================
        // SEE Pruning for noisy moves
        // ====================================================================
        if (!in_singular_search && !is_pv_node && !in_check && is_noisy &&
            !chess::see::see_ge(board, move, 0)) {
            continue;
        }

        // ====================================================================
        // SEE Pruning for quiet moves
        // ====================================================================
        if (!in_singular_search &&
            !is_pv_node &&
            !in_check &&
            depth <= 8 &&
            is_quiet &&
            move != tt_move &&
            !g_killerMoves.is_killer(ply_from_root, move) &&
            move_count >= 2) {

            if (!chess::see::see_ge(board, move, -50 * depth)) {
                continue;
            }
        }

        // ====================================================================
        // History Pruning
        // ====================================================================
        if (!in_singular_search &&
            !is_pv_node &&
            !in_check &&
            depth <= 4 &&
            is_quiet &&
            move_count >= 3 &&
            move != tt_move) {

            int hist_score = g_butterflyHistory.get(side_to_move, move.from(), move.to());
            int threshold = -2000 * depth;
            if (hist_score < threshold) continue;
        }

        // ====================================================================
        // Late Move Pruning (LMP)
        // ====================================================================
        if (!in_singular_search &&
            !is_pv_node &&
            !in_check &&
            depth <= 8 &&
            is_quiet &&
            move != tt_move &&
            move_count >= 5 + 2 * depth * depth) {
            break;
        }

        // ====================================================================
        // Singular Extensions
        // ====================================================================
        int se_ext = 0;
        if (!in_singular_search && move == tt_move && !in_check) {
            auto se = probeSingularExtension(board, depth, beta, ply_from_root,
                                             thread, tm, stats, tt_move, hash,
                                             is_pv_node, is_quiet, ss);
            if (se.multicut) {
                return se.mcScore;
            }
            se_ext = se.ext;
            se_ext = std::min(se_ext, 3);
            se_ext = std::max(se_ext, -1);
        }

        move_count++;

        // Make move
        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);

        ss[ply_from_root].current_move = move;

        bool gives_check = board.inCheck();

        // ====================================================================
        // Futility Pruning
        // ====================================================================
        if (!in_singular_search &&
            !is_pv_node &&
            !in_check &&
            !gives_check &&
            depth <= 7 &&
            is_quiet &&
            move != tt_move &&
            std::abs(alpha) < MATE_SCORE - 100) {

            int futility_margin = 400 + 120 * depth;
            if (static_eval + futility_margin <= alpha) {
                board.unmakeMove(move);
                thread.accumulatorStack.pop();
                continue;
            }
        }

        int eval;
        int local_extension = extension + se_ext;
        int new_depth = depth + local_extension - 1;

        // ====================================================================
        // Late Move Reductions (LMR)
        // ====================================================================
        bool can_reduce = !in_check &&
                         is_quiet &&
                         move_count > 1 &&
                         depth >= 3 &&
                         !gives_check;
        if (in_singular_search) can_reduce = false;

        if (can_reduce) {
            int reduction = lmr_reductions[std::min(depth, 63)][std::min(move_count, 63)];

            // Reduce less for TT move
            if (move == tt_move) reduction = 0;
            // Reduce less for early moves
            else if (move_count <= 3) reduction = std::max(0, reduction - 1);

            reduction = std::max(1, std::min(reduction, new_depth - 1));

            // Reduced depth search (null window)
            eval = -alphaBeta(board, new_depth - reduction, -alpha - 1, -alpha,
                              ply_from_root + 1, thread, tm, stats, true, move, ss);

            // Re-search at full depth if reduced search beats alpha
            if (eval > alpha) {
                eval = -alphaBeta(board, new_depth, -beta, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
            }
        } else {
            // PVS: First move gets full window, rest get null window first
            if (move_count == 1) {
                eval = -alphaBeta(board, new_depth, -beta, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
            } else {
                // Null window search
                eval = -alphaBeta(board, new_depth, -alpha - 1, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
                // Re-search with full window if it beats alpha
                if (eval > alpha && eval < beta) {
                    eval = -alphaBeta(board, new_depth, -beta, -alpha,
                                      ply_from_root + 1, thread, tm, stats, true, move, ss);
                }
            }
        }

        // Unmake move
        board.unmakeMove(move);
        thread.accumulatorStack.pop();

        // Update best
        if (eval > best_score) {
            best_score = eval;
            best_move = move;
        }

        // Beta cutoff
        if (eval >= beta) {
            if (is_quiet) {
                g_killerMoves.store(ply_from_root, move);
                g_counterMoves.update(previous_move, move);

                // History bonus for the move that caused cutoff
                int bonus = 32 * depth * depth;
                g_butterflyHistory.update(side_to_move, move.from(), move.to(), bonus);

                // History malus for quiet moves that didn't cause cutoff
                for (int q = 0; q < quiets_count; ++q) {
                    const auto& quiet = quiets_searched[q];
                    g_butterflyHistory.update(side_to_move, quiet.from(), quiet.to(), -bonus / 2);
                }
            } else {
                // Capture history update for captures that cause cutoff
                chess::Piece captured = board.at(move.to());
                if (captured != chess::Piece::NONE) {
                    int bonus = 32 * depth * depth;
                    int piece_type = static_cast<int>(board.at(move.from()).type());
                    int captured_type = static_cast<int>(captured.type());
                    g_captureHistory.update(piece_type, move.to().index(), captured_type, bonus);
                }
            }
            if (!in_singular_search) {
                storeTT(hash, depth, beta, best_move, TT_LOWER, ply_from_root, tt_pv);
            }
            return beta;
        }

        // Update alpha
        if (eval > alpha) alpha = eval;

        // Track searched quiets (no allocation)
        if (is_quiet && quiets_count < MAX_QUIETS_TRACKED) {
            quiets_searched[quiets_count++] = move;
        }
    }

    // ========================================================================
    // History bonus for best move if it raised alpha
    // ========================================================================
    if (best_score > original_alpha && isQuietMove(board, best_move)) {
        int bonus = 8 * depth * depth;
        g_butterflyHistory.update(side_to_move, best_move.from(), best_move.to(), bonus);
        for (int q = 0; q < quiets_count; ++q) {
            const auto& quiet = quiets_searched[q];
            if (quiet != best_move) {
                g_butterflyHistory.update(side_to_move, quiet.from(), quiet.to(), -bonus / 4);
            }
        }
    }

    // ========================================================================
    // Store TT Entry
    // ========================================================================
    if (!in_singular_search) {
        TTFlag flag = (best_score > original_alpha) ? TT_EXACT : TT_UPPER;
        storeTT(hash, depth, best_score, best_move, flag, ply_from_root, tt_pv);
    }

    return best_score;
}

// ============================================================================
// Iterative Deepening Search
// ============================================================================

chess::Move search(chess::Board& board, int max_depth, ThreadInfo& thread, TimeManager& tm) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) return chess::Move();

    // Single legal move - return immediately
    if (moves.size() == 1) {
        std::cout << "info string only move" << std::endl;
        return moves[0];
    }

    // Age history if it's getting too large
    if (g_butterflyHistory.should_age()) {
        g_butterflyHistory.age();
        std::cout << "info string butterfly history aged" << std::endl;
    }

    chess::Move best_move = moves[0];
    int best_score = -MATE_SCORE;
    double last_depth_ms = 100.0;

    SearchStack ss[MAX_PLY + 10];

    // Root no-alloc scored buffer
    ScoredMove root_scored[MAX_MOVES_NOALLOC];
    int root_scored_count = 0;
    scoreMovesNoAlloc(moves, board, chess::Move(), chess::Move(), board.sideToMove(), 0,
                      root_scored, root_scored_count);

    // ========================================================================
    // Iterative Deepening Loop
    // ========================================================================
    for (int depth = 1; depth <= max_depth; ++depth) {
        auto depth_start = std::chrono::high_resolution_clock::now();

        // Check if we should continue to next depth
        if (depth > 1 && !tm.should_continue_depth(depth, last_depth_ms)) {
            std::cout << "info string stopping at depth " << (depth - 1)
                      << " (time: " << tm.elapsed_ms() << "ms)" << std::endl;
            break;
        }

        SearchStats stats;
        stats.reset();

        int alpha, beta;
        int delta = 25;
        chess::Move depth_best_move = (root_scored_count > 0 ? root_scored[0].move : moves[0]);
        int score;

        // ====================================================================
        // Aspiration Windows
        // ====================================================================
        bool search_again = true;
        while (search_again) {
            search_again = false;

            // Set aspiration window
            if (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100) {
                alpha = std::max(-MATE_SCORE, best_score - delta);
                beta  = std::min( MATE_SCORE, best_score + delta);
            } else {
                alpha = -MATE_SCORE - 1;
                beta  =  MATE_SCORE + 1;
            }

            stats.reset();

            // Rescore root moves with current tt move (best_move)
            scoreMovesNoAlloc(moves, board, best_move, chess::Move(), board.sideToMove(), 0,
                              root_scored, root_scored_count);

            score = -MATE_SCORE;

            // Search each root move
            for (int i = 0; i < root_scored_count; ++i) {
                pickNextMoveNoAlloc(root_scored, root_scored_count, i);
                const auto& move = root_scored[i].move;

                // Time check
                if (tm.should_stop()) {
                    std::cout << "info string time limit reached at depth " << depth << std::endl;
                    goto search_done;
                }

                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);

                // Check for immediate draw
                bool is_draw_move = isDrawByRepetition(board) || isDrawByFiftyMove(board);
                int eval = is_draw_move ? -getDrawScore(1)
                                        : -alphaBeta(board, depth - 1, -beta, -alpha, 1,
                                                     thread, &tm, stats, true, move, ss);

                board.unmakeMove(move);
                thread.accumulatorStack.pop();

                if (eval > score) {
                    score = eval;
                    depth_best_move = move;
                }
                if (eval > alpha) alpha = eval;
                if (eval >= beta) break;
            }

            // Check for aspiration window fail
            int aspiration_alpha = (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100) ?
                                   std::max(-MATE_SCORE, best_score - delta) : -MATE_SCORE - 1;
            int aspiration_beta  = (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100) ?
                                   std::min( MATE_SCORE, best_score + delta) :  MATE_SCORE + 1;

            if (score <= aspiration_alpha) {
                std::cout << "info string depth " << depth << " failed low ("
                          << score << "), widening window" << std::endl;
                delta *= 2;
                search_again = true;
            } else if (score >= aspiration_beta) {
                std::cout << "info string depth " << depth << " failed high ("
                          << score << "), widening window" << std::endl;
                delta *= 2;
                search_again = true;
            }

            // Fall back to full window if delta gets too large
            if (delta > 1000) {
                std::cout << "info string depth " << depth
                          << " window too wide, using full window" << std::endl;
                delta = MATE_SCORE;
            }

            // Time check during re-search
            if (search_again && tm.should_stop()) {
                std::cout << "info string time limit reached during aspiration re-search at depth "
                          << depth << std::endl;
                goto search_done;
            }
        }

        // Update best move and score
        best_score = score;
        best_move = depth_best_move;

        // Calculate timing
        auto depth_end = std::chrono::high_resolution_clock::now();
        last_depth_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            depth_end - depth_start).count();

        int64_t elapsed = tm.elapsed_ms();
        uint64_t nps = (elapsed > 0) ? (stats.nodes * 1000) / elapsed : 0;

        // Extract PV
        auto pv_line = extractPV(board, depth);
        std::string pv_str;
        for (const auto& m : pv_line)
            pv_str += chess::uci::moveToUci(m) + " ";
        if (pv_str.empty())
            pv_str = chess::uci::moveToUci(best_move);

        // Format score
        std::string score_str;
        if (best_score >= MATE_SCORE - 100) {
            int mate_in = (MATE_SCORE - best_score + 1) / 2;
            score_str = "mate " + std::to_string(mate_in);
        } else if (best_score <= -MATE_SCORE + 100) {
            int mate_in = -(MATE_SCORE + best_score) / 2;
            score_str = "mate " + std::to_string(mate_in);
        } else {
            score_str = "cp " + std::to_string(best_score);
        }

        // Output info
        std::cout << "info score " << score_str
                  << " depth " << depth
                  << " nodes " << stats.nodes
                  << " nps " << nps
                  << " time " << elapsed
                  << " pv " << pv_str << std::endl;

        tm.update_stability(best_move);
    }

search_done:
    return best_move;
}