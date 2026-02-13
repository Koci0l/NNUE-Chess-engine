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
#include <chrono>

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

constexpr int SE_MIN_DEPTH = 5;
constexpr int SE_DEPTH_TOL = 3;
constexpr int SE_MARGIN_PER_DEPTH = 2;
constexpr int SE_DOUBLE_BIAS = 55;
constexpr int SE_TRIPLE_BIAS = 120;

constexpr int PROBCUT_MIN_DEPTH = 5;
constexpr int PROBCUT_BETA_MARGIN = 150;
constexpr int PROBCUT_DEPTH_SUBTRACTOR = 4;
constexpr int PROBCUT_IMPROVING_MARGIN = 30;
constexpr int PROBCUT_SEE_THRESHOLD = 100;
constexpr int PROBCUT_MAX_MOVES = 3;

constexpr int SPROBCUT_BETA_MARGIN = 350;
constexpr int SPROBCUT_TT_DEPTH_SUBTRACTOR = 4;

constexpr int MAX_MOVES_NOALLOC = 256;
constexpr int MAX_QUIETS_TRACKED = 64;

constexpr int RAZOR_MARGIN_D1 = 300;
constexpr int RAZOR_MARGIN_D2 = 500;
constexpr int RAZOR_MARGIN_D3 = 700;

// ============================================================================
// Helper Functions
// ============================================================================

inline int scaleNNUE(int raw_score) {
    return raw_score;
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
// No-allocation move scoring + selection
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
                                   int ply_from_root,
                                   const SearchStack* ss) {
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
        chess::Square capSq(move.to().file(), move.from().rank());
        captured = board.at(capSq);
    } else {
        captured = board.at(move.to());
    }
    const bool is_capture = (captured != chess::Piece::NONE);

    if (is_capture || is_promo) {
        int victim = 0;
        if (is_capture) victim = pieceValue(captured.type());

        chess::Piece attackerP = board.at(move.from());
        int attacker = pieceValue(attackerP.type());

        int score = CAPTURE_BONUS + victim * 10 - attacker;

        if (is_promo) {
            score += PROMO_BONUS + pieceValue(move.promotionType());
        }

        return score;
    }

    if (g_killerMoves.is_killer(ply_from_root, move)) return KILLER_BONUS;
    if (move == counter_move) return COUNTER_BONUS;

    chess::Piece cur_piece = board.at(move.from());
    int hist = g_butterflyHistory.get(side_to_move, move.from(), move.to());

    int cont1 = 0, cont2 = 0;
    if (ss != nullptr) {
        if (ply_from_root >= 1 &&
            ss[ply_from_root - 1].moved_piece != chess::Piece::NONE) {
            cont1 = g_contHist1ply.get(ss[ply_from_root - 1].moved_piece,
                                        ss[ply_from_root - 1].current_move.to(),
                                        cur_piece, move.to());
        }
        if (ply_from_root >= 2 &&
            ss[ply_from_root - 2].moved_piece != chess::Piece::NONE) {
            cont2 = g_contHist2ply.get(ss[ply_from_root - 2].moved_piece,
                                        ss[ply_from_root - 2].current_move.to(),
                                        cur_piece, move.to());
        }
    }

    return hist + cont1 + cont2;
}

static inline void scoreMovesNoAlloc(const chess::Movelist& moves,
                                     const chess::Board& board,
                                     chess::Move tt_move,
                                     chess::Move counter_move,
                                     chess::Color side_to_move,
                                     int ply_from_root,
                                     ScoredMove* out,
                                     int& outCount,
                                     const SearchStack* ss) {
    outCount = 0;
    for (const auto& mv : moves) {
        if (outCount >= MAX_MOVES_NOALLOC) break;
        out[outCount].move = mv;
        out[outCount].score = scoreMoveNoAlloc(board, mv, tt_move, counter_move,
                                                side_to_move, ply_from_root, ss);
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
            if (m == move) { is_legal = true; break; }
        }
        if (!is_legal) break;

        pv.push_back(move);
        board.makeMove(move);

        if (board.isRepetition(1) || board.halfMoveClock() >= 100) break;

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

    if (ply_from_root >= MAX_PLY) {
        return scaleNNUE(g_nnue.evaluate(board, thread));
    }

    if (isDrawByRepetition(board) || isDrawByFiftyMove(board)) {
        return getDrawScore(ply_from_root);
    }

    bool in_check = board.inCheck();

    uint64_t hash = getZobristHash(board);
    TTEntry te;
    chess::Move tt_move;
    bool tt_hit = peekTT(hash, te);

    if (tt_hit) {
        tt_move = te.best_move;
        int tt_score = te.score;
        if (tt_score >= MATE_SCORE - 100) tt_score -= ply_from_root;
        else if (tt_score <= -MATE_SCORE + 100) tt_score += ply_from_root;

        if (te.depth >= 0) {
            if (te.flag == TT_EXACT) return tt_score;
            if (te.flag == TT_LOWER && tt_score >= beta) return tt_score;
            if (te.flag == TT_UPPER && tt_score <= alpha) return tt_score;
        }
    }

    int original_alpha = alpha;
    int best_score;

    if (!in_check) {
        best_score = scaleNNUE(g_nnue.evaluate(board, thread));
        if (best_score >= beta) {
            storeTT(hash, 0, best_score, chess::Move(), TT_LOWER, ply_from_root);
            return best_score;
        }
        if (best_score > alpha) alpha = best_score;
    } else {
        best_score = -MATE_SCORE + ply_from_root;
    }

    chess::Movelist all_moves;
    chess::movegen::legalmoves(all_moves, board);

    if (in_check && all_moves.empty()) {
        return -MATE_SCORE + ply_from_root;
    }

    ScoredMove scored_moves[MAX_MOVES_NOALLOC];
    int scored_count = 0;

    if (in_check) {
        scoreMovesNoAlloc(all_moves, board, tt_move, chess::Move(), board.sideToMove(),
                          ply_from_root, scored_moves, scored_count, nullptr);
    } else {
        chess::Movelist tactical_moves;
        for (const auto& move : all_moves) {
            bool is_tactical = board.at(move.to()) != chess::Piece::NONE ||
                               move.typeOf() == chess::Move::PROMOTION ||
                               move.typeOf() == chess::Move::ENPASSANT;
            if (is_tactical) tactical_moves.add(move);
        }
        scoreMovesNoAlloc(tactical_moves, board, tt_move, chess::Move(), board.sideToMove(),
                          ply_from_root, scored_moves, scored_count, nullptr);
    }

    chess::Move best_move;

    for (int i = 0; i < scored_count; ++i) {
        pickNextMoveNoAlloc(scored_moves, scored_count, i);
        const auto& move = scored_moves[i].move;

        bool is_tactical = board.at(move.to()) != chess::Piece::NONE ||
                           move.typeOf() == chess::Move::PROMOTION ||
                           move.typeOf() == chess::Move::ENPASSANT;

        if (!in_check && is_tactical && !chess::see::see_ge(board, move, 0)) continue;

        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);

        int eval = -quiescence(board, -beta, -alpha, thread, ply_from_root + 1, stats);

        board.unmakeMove(move);
        thread.accumulatorStack.pop();

        if (eval > best_score) {
            best_score = eval;
            best_move = move;
        }

        if (eval >= beta) {
            storeTT(hash, 0, best_score, best_move, TT_LOWER, ply_from_root);
            return best_score;
        }

        if (eval > alpha) alpha = eval;
    }

    TTFlag flag = (best_score > original_alpha) ? TT_EXACT : TT_UPPER;
    if (best_move != chess::Move() || !in_check) {
        storeTT(hash, 0, best_score, best_move, flag, ply_from_root);
    }

    return best_score;
}

// ============================================================================
// Continuation History Update Helper
// ============================================================================

static inline void updateContHist(int ply_from_root, const SearchStack* ss,
                                  chess::Piece piece, chess::Square to, int bonus) {
    if (ply_from_root >= 1 &&
        ss[ply_from_root - 1].moved_piece != chess::Piece::NONE) {
        g_contHist1ply.update(ss[ply_from_root - 1].moved_piece,
                              ss[ply_from_root - 1].current_move.to(),
                              piece, to, bonus);
    }
    if (ply_from_root >= 2 &&
        ss[ply_from_root - 2].moved_piece != chess::Piece::NONE) {
        g_contHist2ply.update(ss[ply_from_root - 2].moved_piece,
                              ss[ply_from_root - 2].current_move.to(),
                              piece, to, bonus);
    }
}

static inline int getCombinedHist(chess::Color side, const chess::Move& move,
                                  chess::Piece piece, int ply_from_root,
                                  const SearchStack* ss) {
    int h = g_butterflyHistory.get(side, move.from(), move.to());
    if (ply_from_root >= 1 &&
        ss[ply_from_root - 1].moved_piece != chess::Piece::NONE) {
        h += g_contHist1ply.get(ss[ply_from_root - 1].moved_piece,
                                ss[ply_from_root - 1].current_move.to(),
                                piece, move.to());
    }
    if (ply_from_root >= 2 &&
        ss[ply_from_root - 2].moved_piece != chess::Piece::NONE) {
        h += g_contHist2ply.get(ss[ply_from_root - 2].moved_piece,
                                ss[ply_from_root - 2].current_move.to(),
                                piece, move.to());
    }
    return h;
}

// ============================================================================
// Alpha-Beta Search
// ============================================================================

int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int ply_from_root,
              ThreadInfo& thread, const TimeManager* tm, SearchStats& stats, bool allow_null,
              chess::Move previous_move, SearchStack* ss, chess::Move excluded_move) {
    stats.nodes++;

    if (ply_from_root >= MAX_PLY) {
        return scaleNNUE(g_nnue.evaluate(board, thread));
    }

    if (tm && tm->should_stop()) return alpha;

    if (ply_from_root > 0) {
        if (isDrawByRepetition(board)) return getDrawScore(ply_from_root);
        if (isDrawByFiftyMove(board)) return getDrawScore(ply_from_root);
    }

    bool in_check = board.inCheck();

    if (depth <= 0 && !in_check) {
        return quiescence(board, alpha, beta, thread, ply_from_root, stats);
    }

    bool is_pv_node = (beta - alpha) > 1;
    bool in_singular_search = (excluded_move != chess::Move());

    uint64_t hash = getZobristHash(board);
    int tt_score = 0;
    chess::Move tt_move;
    int tt_depth = 0;
    TTFlag tt_flag = TT_EXACT;
    bool tt_hit = false;

    if (!in_singular_search) {
        TTEntry te;
        tt_hit = peekTT(hash, te);

        if (tt_hit) {
            tt_move = te.best_move;
            tt_depth = te.depth;
            tt_flag = te.flag;

            tt_score = te.score;
            if (tt_score >= MATE_SCORE - 100) tt_score -= ply_from_root;
            else if (tt_score <= -MATE_SCORE + 100) tt_score += ply_from_root;

            if (!is_pv_node && ply_from_root > 0 && te.depth >= depth) {
                if (te.flag == TT_EXACT) return tt_score;
                if (te.flag == TT_LOWER && tt_score >= beta) return tt_score;
                if (te.flag == TT_UPPER && tt_score <= alpha) return tt_score;
            }
        }
    } else {
        tt_move = chess::Move();
    }

    // IID
    if (!in_singular_search && !is_pv_node && depth >= 4 && tt_move == chess::Move() && !in_check) {
        depth--;
    }

    int static_eval = 0;
    bool improving = false;

    if (!in_check) {
        static_eval = scaleNNUE(g_nnue.evaluate(board, thread));

        // TT-based eval correction
        if (tt_hit && !in_singular_search && std::abs(tt_score) < MATE_SCORE - 100) {
            if (tt_flag == TT_EXACT ||
                (tt_flag == TT_LOWER && tt_score > static_eval) ||
                (tt_flag == TT_UPPER && tt_score < static_eval)) {
                static_eval = tt_score;
            }
        }

        ss[ply_from_root].static_eval = static_eval;
        if (ply_from_root >= 2 && ss[ply_from_root - 2].static_eval != -MATE_SCORE) {
            improving = ss[ply_from_root].static_eval > ss[ply_from_root - 2].static_eval;
        }
    } else {
        ss[ply_from_root].static_eval = -MATE_SCORE;
    }

    int extension = in_check ? 1 : 0;

    // RFP
    if (!is_pv_node && !in_check && !in_singular_search &&
        depth <= 7 && depth >= 1 && std::abs(beta) < MATE_SCORE - 100) {
        int rfp_margin = 85 * depth;
        if (static_eval - rfp_margin >= beta) {
            return static_eval - rfp_margin;
        }
    }

    // Razoring
    if (!is_pv_node && !in_check && !in_singular_search && depth <= 3) {
        int razor_margin = (depth == 1) ? RAZOR_MARGIN_D1 :
                           (depth == 2) ? RAZOR_MARGIN_D2 : RAZOR_MARGIN_D3;

        if (static_eval + razor_margin <= alpha) {
            int razor_score = quiescence(board, alpha, beta, thread, ply_from_root, stats);
            if (razor_score <= alpha) {
                return razor_score;
            }
        }
    }

    // NMP
    if (allow_null && !in_check && !is_pv_node && !in_singular_search &&
        depth >= 3 && hasNonPawnMaterial(board) && static_eval >= beta) {

        int R = 3 + depth / 3;
        R = std::min(R, depth - 1);

        AccumulatorPair saved_acc = thread.accumulatorStack.current();

        board.makeNullMove();
        thread.accumulatorStack.push();
        thread.accumulatorStack.current() = saved_acc;

        ss[ply_from_root].current_move = chess::Move();
        ss[ply_from_root].moved_piece = chess::Piece::NONE;

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

    // ProbCut
    int probcut_beta = beta + PROBCUT_BETA_MARGIN - PROBCUT_IMPROVING_MARGIN * (int)improving;

    if (!in_singular_search && !is_pv_node && !in_check &&
        depth >= PROBCUT_MIN_DEPTH && std::abs(beta) < MATE_SCORE - 200 &&
        hasNonPawnMaterial(board)) {

        bool tt_gate = true;
        if (tt_move != chess::Move() && tt_depth >= depth - 3 && tt_score < probcut_beta) {
            tt_gate = false;
        }

        if (tt_gate) {
            int probcut_depth = depth - PROBCUT_DEPTH_SUBTRACTOR;

            chess::Movelist all_moves;
            chess::movegen::legalmoves(all_moves, board);

            struct ProbCutMove { chess::Move mv; int score; };
            std::vector<ProbCutMove> probcut_moves;
            probcut_moves.reserve(all_moves.size());

            for (const auto& mv : all_moves) {
                bool is_capture = board.at(mv.to()) != chess::Piece::NONE ||
                                  mv.typeOf() == chess::Move::ENPASSANT;
                bool is_promotion = mv.typeOf() == chess::Move::PROMOTION;
                if (!is_capture && !is_promotion) continue;
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

            std::sort(probcut_moves.begin(), probcut_moves.end(),
                      [](const ProbCutMove& a, const ProbCutMove& b) { return a.score > b.score; });

            for (size_t i = 0; i < std::min(probcut_moves.size(), size_t(PROBCUT_MAX_MOVES)); ++i) {
                const auto& move = probcut_moves[i].mv;

                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);

                int probcut_value = -quiescence(board, -probcut_beta, -probcut_beta + 1,
                                               thread, ply_from_root + 1, stats);

                if (probcut_value >= probcut_beta) {
                    probcut_value = -alphaBeta(board, probcut_depth - 1,
                                               -probcut_beta, -probcut_beta + 1,
                                               ply_from_root + 1, thread, tm, stats, false, move, ss);
                }

                board.unmakeMove(move);
                thread.accumulatorStack.pop();

                if (probcut_value >= probcut_beta) {
                    storeTT(hash, probcut_depth, probcut_value, move, TT_LOWER, ply_from_root);
                    return probcut_value;
                }

                if (tm && tm->should_stop()) break;
            }
        }
    }

    // Small ProbCut
    int small_probcut_beta = beta + SPROBCUT_BETA_MARGIN;
    if (!in_singular_search && !is_pv_node && tt_move != chess::Move() &&
        tt_flag == TT_LOWER && tt_depth >= depth - SPROBCUT_TT_DEPTH_SUBTRACTOR &&
        tt_score >= small_probcut_beta && std::abs(tt_score) < MATE_SCORE - 200 &&
        std::abs(beta) < MATE_SCORE - 200) {
        return small_probcut_beta;
    }

    // Move generation
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        if (in_check) return -MATE_SCORE + ply_from_root;
        return getDrawScore(ply_from_root);
    }

    chess::Move counter_move = g_counterMoves.get(previous_move);
    chess::Color side_to_move = board.sideToMove();

    ScoredMove scored_moves[MAX_MOVES_NOALLOC];
    int scored_count = 0;
    scoreMovesNoAlloc(moves, board, tt_move, counter_move, side_to_move, ply_from_root,
                      scored_moves, scored_count, ss);

    chess::Move best_move;
    int best_score = -MATE_SCORE;
    int original_alpha = alpha;

    chess::Move quiets_searched[MAX_QUIETS_TRACKED];
    int quiets_count = 0;
    int move_count = 0;

    // Move loop
    for (int i = 0; i < scored_count; ++i) {
        pickNextMoveNoAlloc(scored_moves, scored_count, i);
        const auto& move = scored_moves[i].move;

        if (move == excluded_move) continue;

        bool is_quiet = isQuietMove(board, move);
        bool is_noisy = !is_quiet;

        // SEE pruning noisy
        if (!in_singular_search && !is_pv_node && !in_check && is_noisy &&
            best_score > -MATE_SCORE + 100 && !chess::see::see_ge(board, move, 0)) {
            continue;
        }

        // SEE pruning quiet
        if (!in_singular_search && !is_pv_node && !in_check && depth <= 8 &&
            is_quiet && move != tt_move && best_score > -MATE_SCORE + 100 &&
            !g_killerMoves.is_killer(ply_from_root, move) && move_count >= 2) {
            if (!chess::see::see_ge(board, move, -50 * depth)) {
                continue;
            }
        }

        // History pruning
        if (!in_singular_search && !is_pv_node && !in_check && depth <= 4 &&
            is_quiet && move_count >= 3 && move != tt_move && best_score > -MATE_SCORE + 100) {
            chess::Piece hp = board.at(move.from());
            int hist_score = getCombinedHist(side_to_move, move, hp, ply_from_root, ss);
            if (hist_score < -2000 * depth) continue;
        }

        // LMP
        if (!in_singular_search && !is_pv_node && !in_check && depth <= 8 &&
            is_quiet && move != tt_move && best_score > -MATE_SCORE + 100 &&
            move_count >= 5 + 2 * depth * depth) {
            break;
        }

        // Singular extensions
        int se_ext = 0;
        if (!in_singular_search && move == tt_move && !in_check) {
            auto se = probeSingularExtension(board, depth, beta, ply_from_root,
                                             thread, tm, stats, tt_move, hash,
                                             is_pv_node, is_quiet, ss);
            if (se.multicut) return se.mcScore;
            se_ext = std::clamp(se.ext, -1, 3);
        }

        move_count++;

        chess::Piece moved_piece = board.at(move.from());

        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);

        ss[ply_from_root].current_move = move;
        ss[ply_from_root].moved_piece = moved_piece;
        bool gives_check = board.inCheck();

        // Futility pruning
        if (!in_singular_search && !is_pv_node && !in_check && !gives_check &&
            depth <= 7 && is_quiet && move != tt_move && best_score > -MATE_SCORE + 100 &&
            std::abs(alpha) < MATE_SCORE - 100) {
            int futility_margin = 100 + 80 * depth;
            if (static_eval + futility_margin <= alpha) {
                board.unmakeMove(move);
                thread.accumulatorStack.pop();
                continue;
            }
        }

        int eval;
        int local_extension = extension + se_ext;
        int new_depth = depth + local_extension - 1;

        // LMR
        bool can_reduce = !in_check && is_quiet && move_count > 1 &&
                          depth >= 3 && !gives_check && !in_singular_search;

        if (can_reduce) {
            int reduction = lmr_reductions[std::min(depth, 63)][std::min(move_count, 63)];

            if (move == tt_move) reduction = 0;
            else if (move_count <= 3) reduction = std::max(0, reduction - 1);

            if (!is_pv_node) reduction += 1;
            if (!improving) reduction += 1;

            int combined_hist = getCombinedHist(side_to_move, move, moved_piece,
                                                ply_from_root, ss);
            if (combined_hist < -4000) reduction += 1;
            else if (combined_hist > 8000) reduction = std::max(0, reduction - 1);

            reduction = std::clamp(reduction, 1, new_depth - 1);

            eval = -alphaBeta(board, new_depth - reduction, -alpha - 1, -alpha,
                              ply_from_root + 1, thread, tm, stats, true, move, ss);

            if (eval > alpha) {
                eval = -alphaBeta(board, new_depth, -beta, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
            }
        } else {
            if (move_count == 1) {
                eval = -alphaBeta(board, new_depth, -beta, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
            } else {
                eval = -alphaBeta(board, new_depth, -alpha - 1, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
                if (eval > alpha && eval < beta) {
                    eval = -alphaBeta(board, new_depth, -beta, -alpha,
                                      ply_from_root + 1, thread, tm, stats, true, move, ss);
                }
            }
        }

        board.unmakeMove(move);
        thread.accumulatorStack.pop();

        if (eval > best_score) {
            best_score = eval;
            best_move = move;
        }

        if (eval >= beta) {
            if (is_quiet) {
                g_killerMoves.store(ply_from_root, move);
                g_counterMoves.update(previous_move, move);

                int bonus = 32 * depth * depth;
                g_butterflyHistory.update(side_to_move, move.from(), move.to(), bonus);

                chess::Piece cut_piece = board.at(move.from());
                updateContHist(ply_from_root, ss, cut_piece, move.to(), bonus);

                for (int q = 0; q < quiets_count; ++q) {
                    g_butterflyHistory.update(side_to_move, quiets_searched[q].from(),
                                            quiets_searched[q].to(), -bonus / 2);

                    chess::Piece qp = board.at(quiets_searched[q].from());
                    updateContHist(ply_from_root, ss, qp, quiets_searched[q].to(), -bonus / 2);
                }
            } else {
                chess::Piece captured = board.at(move.to());
                if (captured != chess::Piece::NONE) {
                    int bonus = 32 * depth * depth;
                    g_captureHistory.update(static_cast<int>(board.at(move.from()).type()),
                                        move.to().index(), static_cast<int>(captured.type()), bonus);
                }
            }
            if (!in_singular_search) {
                storeTT(hash, depth, beta, best_move, TT_LOWER, ply_from_root);
            }
            return beta;
        }

        if (eval > alpha) alpha = eval;

        if (is_quiet && quiets_count < MAX_QUIETS_TRACKED) {
            quiets_searched[quiets_count++] = move;
        }
    }

    if (best_score > original_alpha && best_move != chess::Move() && isQuietMove(board, best_move)) {
        int bonus = 8 * depth * depth;
        g_butterflyHistory.update(side_to_move, best_move.from(), best_move.to(), bonus);

        chess::Piece pv_piece = board.at(best_move.from());
        updateContHist(ply_from_root, ss, pv_piece, best_move.to(), bonus);

        for (int q = 0; q < quiets_count; ++q) {
            if (quiets_searched[q] != best_move) {
                g_butterflyHistory.update(side_to_move, quiets_searched[q].from(),
                                          quiets_searched[q].to(), -bonus / 4);

                chess::Piece qp = board.at(quiets_searched[q].from());
                updateContHist(ply_from_root, ss, qp, quiets_searched[q].to(), -bonus / 4);
            }
        }
    }

    if (!in_singular_search) {
        TTFlag flag = (best_score > original_alpha) ? TT_EXACT : TT_UPPER;
        storeTT(hash, depth, best_score, best_move, flag, ply_from_root);
    }

    return best_score;
}

// ============================================================================
// Iterative Deepening
// ============================================================================

chess::Move search(chess::Board& board, int max_depth, ThreadInfo& thread, TimeManager& tm,
                   int64_t node_limit) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) return chess::Move();
    if (moves.size() == 1) {
        std::cout << "info string only move" << std::endl;
        return moves[0];
    }

    if (g_butterflyHistory.should_age()) {
        g_butterflyHistory.age();
        g_contHist1ply.age();
        g_contHist2ply.age();
    }

    chess::Move best_move = moves[0];
    int best_score = -MATE_SCORE;
    double last_depth_ms = 100.0;

    SearchStack ss[MAX_PLY + 10];
    for (int i = 0; i < MAX_PLY + 10; ++i) {
        ss[i].static_eval = -MATE_SCORE;
        ss[i].current_move = chess::Move();
        ss[i].moved_piece = chess::Piece::NONE;
    }

    ScoredMove root_scored[MAX_MOVES_NOALLOC];
    int root_scored_count = 0;

    SearchStats stats;
    stats.reset();

    if (node_limit > 0) {
        tm.set_node_limit(node_limit, &stats.nodes);
    }

    for (int depth = 1; depth <= max_depth; ++depth) {
        auto depth_start = std::chrono::high_resolution_clock::now();

        if (depth > 1 && !tm.should_continue_depth(depth, last_depth_ms)) {
            break;
        }

        chess::Move depth_best_move = best_move;
        int score = -MATE_SCORE;
        int delta = 25;
        int aspiration_alpha, aspiration_beta;

        while (true) {
            if (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100) {
                aspiration_alpha = std::max(-MATE_SCORE, best_score - delta);
                aspiration_beta  = std::min( MATE_SCORE, best_score + delta);
            } else {
                aspiration_alpha = -MATE_SCORE;
                aspiration_beta  =  MATE_SCORE;
            }

            int alpha = aspiration_alpha;
            int beta = aspiration_beta;

            score = -MATE_SCORE;
            depth_best_move = best_move;

            scoreMovesNoAlloc(moves, board, best_move, chess::Move(), board.sideToMove(), 0,
                              root_scored, root_scored_count, ss);

            for (int i = 0; i < root_scored_count; ++i) {
                pickNextMoveNoAlloc(root_scored, root_scored_count, i);
                const auto& move = root_scored[i].move;

                if (tm.should_stop()) goto search_done;

                chess::Piece root_piece = board.at(move.from());

                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);

                ss[0].current_move = move;
                ss[0].moved_piece = root_piece;

                int eval;
                bool is_draw_move = isDrawByRepetition(board) || isDrawByFiftyMove(board);

                if (is_draw_move) {
                    eval = -getDrawScore(1);
                } else if (i == 0) {
                    eval = -alphaBeta(board, depth - 1, -beta, -alpha, 1,
                                     thread, &tm, stats, true, move, ss);
                } else {
                    eval = -alphaBeta(board, depth - 1, -alpha - 1, -alpha, 1,
                                     thread, &tm, stats, true, move, ss);
                    if (eval > alpha && eval < beta) {
                        eval = -alphaBeta(board, depth - 1, -beta, -alpha, 1,
                                         thread, &tm, stats, true, move, ss);
                    }
                }

                board.unmakeMove(move);
                thread.accumulatorStack.pop();

                if (eval > score) {
                    score = eval;
                    depth_best_move = move;
                }
                if (eval > alpha) alpha = eval;
                if (eval >= beta) break;
            }

            if (score <= aspiration_alpha && aspiration_alpha > -MATE_SCORE) {
                delta *= 2;
                best_score = score;
                if (delta > 500) delta = MATE_SCORE;
            } else if (score >= aspiration_beta && aspiration_beta < MATE_SCORE) {
                delta *= 2;
                best_score = score;
                best_move = depth_best_move;
                if (delta > 500) delta = MATE_SCORE;
            } else {
                break;
            }

            if (tm.should_stop()) goto search_done;
        }

        best_score = score;
        best_move = depth_best_move;

        auto depth_end = std::chrono::high_resolution_clock::now();
        last_depth_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            depth_end - depth_start).count();

        int64_t elapsed = tm.elapsed_ms();
        int64_t elapsed_for_nps = std::max<int64_t>(1, elapsed);
        uint64_t nps = (stats.nodes * 1000) / elapsed_for_nps;

        auto pv_line = extractPV(board, depth);
        std::string pv_str;
        for (const auto& m : pv_line) pv_str += chess::uci::moveToUci(m) + " ";
        if (pv_str.empty()) pv_str = chess::uci::moveToUci(best_move);

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

        std::cout << "info score " << score_str
                  << " depth " << depth
                  << " nodes " << stats.nodes
                  << " nps " << nps
                  << " time " << elapsed
                  << " pv " << pv_str << std::endl;

        tm.update_stability(best_move);
    }

search_done:
    std::cerr << "info string total nodes searched " << stats.nodes << std::endl;
    return best_move;
}