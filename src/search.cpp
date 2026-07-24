#include "search.h"
#include "tt.h"
#include "history.h"
#include "movepick.h"
#include "see.h"
#include "zobrist.h"
#include "policy.h"
#include "policy.h"
#include "policy_tune.h"
#include "policy_tune.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

bool g_silent = false;

static int lmr_reductions[64][64];

// Global policy cache (separate from TT)
static PolicyCache g_policy_cache;

void initLMR() {
    for (int depth = 1; depth < 64; ++depth) {
        for (int move_num = 1; move_num < 64; ++move_num) {
            double reduction = 0.75 + std::log(depth) * std::log(move_num) / 2.25;
            lmr_reductions[depth][move_num] = static_cast<int>(reduction);
        }
    }
}

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

constexpr int MAX_QUIETS_TRACKED = 64;
constexpr int MAX_CAPTURES_TRACKED = 32;

constexpr int RAZOR_MARGIN_D1 = 300;
constexpr int RAZOR_MARGIN_D2 = 500;
constexpr int RAZOR_MARGIN_D3 = 700;

inline int scaleNNUE(int raw_score) {
    return raw_score;
}

static inline int correctedEval(int raw_eval, chess::Color side, uint64_t hash) {
    return std::clamp(raw_eval + g_correctionHistory.get(side, hash),
                      -MATE_SCORE + 1, MATE_SCORE - 1);
}

static inline void updateCorrection(chess::Color side, uint64_t hash,
                                    int depth, int raw_static_eval, int score) {
    if (depth < 4) return;
    if (std::abs(raw_static_eval) >= MATE_SCORE - 200) return;
    if (std::abs(score) >= MATE_SCORE - 200) return;
    int diff = std::clamp(score - raw_static_eval, -64, 64);
    g_correctionHistory.update(side, hash, diff, depth);
}

bool isDrawByRepetition(const chess::Board& board) {
    return board.isRepetition(1);
}

bool isDrawByFiftyMove(const chess::Board& board) {
    return board.halfMoveClock() >= 100;
}

int getDrawScore(int) {
    return CONTEMPT;
}

struct CaptureSearchInfo {
    chess::Move move;
    int piece_type;
    int to_sq;
    int captured_type;
};

static inline bool extractCaptureInfo(const chess::Board& board,
                                      const chess::Move& move,
                                      CaptureSearchInfo& info) {
    chess::Piece attacker = board.at(move.from());
    chess::Piece captured;
    if (move.typeOf() == chess::Move::ENPASSANT) {
        chess::Square capSq(move.to().file(), move.from().rank());
        captured = board.at(capSq);
    } else {
        captured = board.at(move.to());
    }
    if (captured == chess::Piece::NONE) return false;
    info.move = move;
    info.piece_type = static_cast<int>(attacker.type());
    info.to_sq = move.to().index();
    info.captured_type = static_cast<int>(captured.type());
    return true;
}

void updateAccumulatorForMove(AccumulatorStack& accStack, chess::Board& board,
                              const chess::Move& move) {
    auto moveType = move.typeOf();
    if (moveType == chess::Move::NORMAL) {
        chess::Piece piece = board.at(move.from());
        chess::Piece captured = board.at(move.to());
        if (captured != chess::Piece::NONE)
            accStack.current().remove_piece(captured, move.to());
        accStack.current().move_piece(piece, move.from(), move.to());
    } else if (moveType == chess::Move::PROMOTION) {
        chess::Piece pawn = board.at(move.from());
        chess::Piece captured = board.at(move.to());
        chess::Piece promotedPiece = chess::Piece(move.promotionType(), pawn.color());
        if (captured != chess::Piece::NONE)
            accStack.current().remove_piece(captured, move.to());
        accStack.current().remove_piece(pawn, move.from());
        accStack.current().add_piece(promotedPiece, move.to());
    } else if (moveType == chess::Move::ENPASSANT) {
        chess::Piece pawn = board.at(move.from());
        chess::Square capturedPawnSq(move.to().file(), move.from().rank());
        chess::Piece capturedPawn = board.at(capturedPawnSq);
        accStack.current().remove_piece(capturedPawn, capturedPawnSq);
        accStack.current().move_piece(pawn, move.from(), move.to());
    } else if (moveType == chess::Move::CASTLING) {
        chess::Square king_from = move.from();
        chess::Square rook_from = move.to();
        bool king_side = rook_from > king_from;
        chess::Color c = board.at(king_from).color();
        chess::Square king_to = chess::Square::castling_king_square(king_side, c);
        chess::Square rook_to = chess::Square::castling_rook_square(king_side, c);
        chess::Piece king_piece = chess::Piece(chess::PieceType::KING, c);
        chess::Piece rook_piece = chess::Piece(chess::PieceType::ROOK, c);
        accStack.current().remove_piece(king_piece, king_from);
        accStack.current().remove_piece(rook_piece, rook_from);
        accStack.current().add_piece(king_piece, king_to);
        accStack.current().add_piece(rook_piece, rook_to);
    }
}

struct SEResult {
    int ext = 0;
    bool multicut = false;
    int mcScore = 0;
};

static inline void updateContHist(int ply_from_root, const SearchStack* ss,
                                  chess::Piece piece, chess::Square to, int bonus) {
    if (ply_from_root >= 1 && ss[ply_from_root - 1].moved_piece != chess::Piece::NONE)
        g_contHist1ply.update(ss[ply_from_root - 1].moved_piece,
                              ss[ply_from_root - 1].current_move.to(), piece, to, bonus);
    if (ply_from_root >= 2 && ss[ply_from_root - 2].moved_piece != chess::Piece::NONE)
        g_contHist2ply.update(ss[ply_from_root - 2].moved_piece,
                              ss[ply_from_root - 2].current_move.to(), piece, to, bonus);
}

static inline int getCombinedHist(chess::Color side, const chess::Move& move,
                                  chess::Piece piece, int ply_from_root,
                                  const SearchStack* ss) {
    int h = g_butterflyHistory.get(side, move.from(), move.to());
    if (ply_from_root >= 1 && ss[ply_from_root - 1].moved_piece != chess::Piece::NONE)
        h += g_contHist1ply.get(ss[ply_from_root - 1].moved_piece,
                                ss[ply_from_root - 1].current_move.to(), piece, move.to());
    if (ply_from_root >= 2 && ss[ply_from_root - 2].moved_piece != chess::Piece::NONE)
        h += g_contHist2ply.get(ss[ply_from_root - 2].moved_piece,
                                ss[ply_from_root - 2].current_move.to(), piece, move.to());
    return h;
}

SEResult probeSingularExtension(chess::Board& board, int depth, int beta, int ply_from_root,
                                ThreadInfo& thread, const TimeManager* tm, SearchStats& stats,
                                chess::Move tt_move, uint64_t hash,
                                bool is_pv_node, bool is_quiet_move, SearchStack* ss) {
    SEResult out;
    if (depth < SE_MIN_DEPTH || tt_move == chess::Move()) return out;
    TTEntry te;
    if (!peekTT(hash, te)) return out;
    if (chess::Move(te.best_move) != tt_move) return out;
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

std::vector<chess::Move> extractPV(chess::Board board, int max_depth) {
    std::vector<chess::Move> pv;
    for (int i = 0; i < max_depth; ++i) {
        TTEntry entry;
        if (!peekTT(getZobristHash(board), entry) || entry.best_move == 0) break;
        chess::Move move = chess::Move(entry.best_move);
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
// Quiescence search
// ============================================================================

int quiescence(chess::Board& board, int alpha, int beta,
               ThreadInfo& thread, int ply_from_root, SearchStats& stats) {
    stats.nodes++;
    if (ply_from_root >= MAX_PLY)
        return scaleNNUE(g_nnue.evaluate(board, thread));
    if (isDrawByRepetition(board) || isDrawByFiftyMove(board))
        return getDrawScore(ply_from_root);

    bool in_check = board.inCheck();
    uint64_t hash = getZobristHash(board);

    TTEntry te;
    chess::Move tt_move = chess::Move();
    bool tt_hit = peekTT(hash, te);
    if (tt_hit) {
        tt_move = chess::Move(te.best_move);
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

    QSearchMovePicker mp(board, tt_move, in_check);
    chess::Move best_move;
    bool searched_any = false;

    while (true) {
        chess::Move move = mp.next();
        if (move == chess::Move()) break;
        searched_any = true;

        bool is_tactical = board.at(move.to()) != chess::Piece::NONE ||
                           move.typeOf() == chess::Move::PROMOTION ||
                           move.typeOf() == chess::Move::ENPASSANT;

        if (!in_check && is_tactical && !chess::see::see_ge(board, move, 0))
            continue;

        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);

        int eval = -quiescence(board, -beta, -alpha, thread, ply_from_root + 1, stats);

        board.unmakeMove(move);
        thread.accumulatorStack.pop();

        if (eval > best_score) { best_score = eval; best_move = move; }
        if (eval >= beta) {
            storeTT(hash, 0, best_score, best_move, TT_LOWER, ply_from_root);
            return best_score;
        }
        if (eval > alpha) alpha = eval;
    }

    if (in_check && !searched_any)
        return -MATE_SCORE + ply_from_root;

    TTFlag flag = (best_score > original_alpha) ? TT_EXACT : TT_UPPER;
    if (best_move != chess::Move() || !in_check)
        storeTT(hash, 0, best_score, best_move, flag, ply_from_root);
    return best_score;
}

// ============================================================================
// Alpha-Beta with full policy integration
// ============================================================================

int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int ply_from_root,
              ThreadInfo& thread, const TimeManager* tm, SearchStats& stats, bool allow_null,
              chess::Move previous_move, SearchStack* ss, chess::Move excluded_move) {
    stats.nodes++;

    alpha = std::max(alpha, -MATE_SCORE + ply_from_root);
    beta = std::min(beta, MATE_SCORE - ply_from_root - 1);
    if (alpha >= beta) return alpha;

    if (ply_from_root >= MAX_PLY)
        return scaleNNUE(g_nnue.evaluate(board, thread));
    if (tm && tm->should_stop()) return alpha;

    if (ply_from_root > 0) {
        if (isDrawByRepetition(board)) return getDrawScore(ply_from_root);
        if (isDrawByFiftyMove(board)) return getDrawScore(ply_from_root);
    }

    bool in_check = board.inCheck();

    if (depth <= 0 && !in_check)
        return quiescence(board, alpha, beta, thread, ply_from_root, stats);

    bool is_pv_node = (beta - alpha) > 1;
    bool in_singular_search = (excluded_move != chess::Move());
    uint64_t hash = getZobristHash(board);

    int tt_score = 0;
    chess::Move tt_move = chess::Move();
    int tt_depth = 0;
    TTFlag tt_flag = TT_EXACT;
    bool tt_hit = false;

    if (!in_singular_search) {
        TTEntry te;
        tt_hit = peekTT(hash, te);
        if (tt_hit) {
            tt_move = chess::Move(te.best_move);
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
    }

    if (!in_singular_search && !is_pv_node && depth >= 4 &&
        tt_move == chess::Move() && !in_check) {
        depth--;
    }

    int raw_static_eval = 0;
    int static_eval = 0;
    bool improving = false;

    if (!in_check) {
        raw_static_eval = scaleNNUE(g_nnue.evaluate(board, thread));
        static_eval = correctedEval(raw_static_eval, board.sideToMove(), hash);
        if (tt_hit && !in_singular_search && std::abs(tt_score) < MATE_SCORE - 100) {
            if (tt_flag == TT_EXACT ||
                (tt_flag == TT_LOWER && tt_score > static_eval) ||
                (tt_flag == TT_UPPER && tt_score < static_eval)) {
                static_eval = tt_score;
            }
        }
        ss[ply_from_root].static_eval = static_eval;
        if (ply_from_root >= 2 && ss[ply_from_root - 2].static_eval != -MATE_SCORE)
            improving = ss[ply_from_root].static_eval > ss[ply_from_root - 2].static_eval;
    } else {
        ss[ply_from_root].static_eval = -MATE_SCORE;
    }

    int extension = in_check ? 1 : 0;

    // RFP
    int rfp_margin = (improving ? 70 : 95) * depth;
    if (!is_pv_node && !in_check && !in_singular_search &&
        depth <= 7 && depth >= 1 && std::abs(beta) < MATE_SCORE - 100) {
        if (static_eval - rfp_margin >= beta)
            return static_eval - rfp_margin;
    }

    // Razoring
    if (!is_pv_node && !in_check && !in_singular_search && depth <= 3) {
        int razor_margin = (depth == 1) ? RAZOR_MARGIN_D1 :
                           (depth == 2) ? RAZOR_MARGIN_D2 : RAZOR_MARGIN_D3;
        if (static_eval + razor_margin <= alpha) {
            int razor_score = quiescence(board, alpha, beta, thread, ply_from_root, stats);
            if (razor_score <= alpha) return razor_score;
        }
    }

    // NMP
    if (allow_null && !in_check && !is_pv_node && !in_singular_search &&
        depth >= 3 && hasNonPawnMaterial(board) && static_eval >= beta) {
        int R = 3 + depth / 3 + (improving ? 1 : 0);
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

    // ========================================================================
    // POLICY: Compute advice at this node
    // ========================================================================

    PolicyAdvice policy_advice;
    bool use_policy_node = false;

    bool any_internal_policy =
        policyWeight(g_pt.internal_policy_order) > 0.0f ||
        policyWeight(g_pt.internal_policy_lmr) > 0.0f ||
        policyWeight(g_pt.policy_top_protect_lmr) > 0.0f ||
        policyWeight(g_pt.policy_tail_reduce) > 0.0f ||
        policyWeight(g_pt.policy_lmp) > 0.0f ||
        policyWeight(g_pt.policy_whitelist) > 0.0f ||
        policyWeight(g_pt.policy_cumulative_mass) > 0.0f ||
        policyWeight(g_pt.policy_futility) > 0.0f ||
        policyWeight(g_pt.policy_see) > 0.0f ||
        policyWeight(g_pt.policy_history_prune) > 0.0f ||
        policyWeight(g_pt.policy_protect) > 0.0f ||
        policyWeight(g_pt.policy_hard_prune) > 0.0f ||
        policyWeight(g_pt.policy_extension) > 0.0f ||
        policyWeight(g_pt.policy_negative_extension) > 0.0f ||
        policyWeight(g_pt.policy_iid) > 0.0f ||
        policyWeight(g_pt.policy_se_gate) > 0.0f ||
        policyWeight(g_pt.policy_probcut) > 0.0f ||
        policyWeight(g_pt.policy_sacrifice) > 0.0f ||
        policyWeight(g_pt.policy_bad_capture) > 0.0f;

    if (any_internal_policy &&
        g_policy.loaded &&
        !in_check &&
        !in_singular_search &&
        depth >= 5 &&
        ply_from_root < 80 &&
        (!is_pv_node || depth >= 7) &&
        std::abs(beta) < MATE_SCORE - 200) {

        use_policy_node = true;

        if (!g_policy_cache.get(hash, policy_advice)) {
            if (computePolicyAdvice(board, policy_advice)) {
                g_policy_cache.put(hash, policy_advice);
            } else {
                use_policy_node = false;
            }
        }
    }

    // ========================================================================
    // POLICY: IID — use policy top as pseudo-TT when no TT move
    // ========================================================================

    float iid_w = policyWeight(g_pt.policy_iid);

    if (iid_w > 0.0f && use_policy_node && policy_advice.ok &&
        tt_move == chess::Move() && policy_advice.sharpness > 0.35f) {
        float required_prob = 0.45f - 0.20f * iid_w;
        if (policy_advice.top_prob > required_prob) {
            tt_move = policy_advice.top_move;
        }
    }

    // ========================================================================
    // ProbCut (captures)
    // ========================================================================

    int probcut_beta = beta + PROBCUT_BETA_MARGIN - PROBCUT_IMPROVING_MARGIN * int(improving);

    if (!in_singular_search && !is_pv_node && !in_check &&
        depth >= PROBCUT_MIN_DEPTH && std::abs(beta) < MATE_SCORE - 200 &&
        hasNonPawnMaterial(board)) {
        bool tt_gate = true;
        if (tt_move != chess::Move() && tt_depth >= depth - 3 && tt_score < probcut_beta)
            tt_gate = false;

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
                if (mv.typeOf() == chess::Move::ENPASSANT)
                    victim = pieceValue(chess::PieceType::PAWN);
                else if (board.at(mv.to()) != chess::Piece::NONE)
                    victim = pieceValue(board.at(mv.to()).type());
                int attacker = pieceValue(board.at(mv.from()).type());
                int move_score = victim * 10 - attacker + (is_promotion ? 500 : 0);
                probcut_moves.push_back({mv, move_score});
            }

            std::sort(probcut_moves.begin(), probcut_moves.end(),
                      [](const ProbCutMove& a, const ProbCutMove& b) {
                          return a.score > b.score;
                      });

            for (size_t i = 0; i < std::min(probcut_moves.size(), size_t(PROBCUT_MAX_MOVES)); ++i) {
                const auto& move = probcut_moves[i].mv;
                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);
                int probcut_value = -quiescence(board, -probcut_beta, -probcut_beta + 1,
                                                thread, ply_from_root + 1, stats);
                if (probcut_value >= probcut_beta) {
                    probcut_value = -alphaBeta(board, probcut_depth,
                                               -probcut_beta, -probcut_beta + 1,
                                               ply_from_root + 1, thread, tm, stats, false,
                                               move, ss);
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

    // ========================================================================
    // POLICY: Quiet ProbCut
    // ========================================================================

    float probcut_quiet_w = policyWeight(g_pt.policy_probcut);

    if (probcut_quiet_w > 0.0f && use_policy_node && policy_advice.ok &&
        !is_pv_node && !in_check && depth >= 6 &&
        static_eval >= beta - 120 &&
        policy_advice.top_prob > 0.35f &&
        policy_advice.sharpness > 0.40f &&
        isQuietMove(board, policy_advice.top_move) &&
        policy_advice.top_move != tt_move &&
        policy_advice.top_move != excluded_move) {

        int pc_beta = beta + int(80.0f * probcut_quiet_w);
        int pc_depth = depth - 4;

        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, policy_advice.top_move);
        board.makeMove(policy_advice.top_move);

        int pc_val = -quiescence(board, -pc_beta, -pc_beta + 1,
                                 thread, ply_from_root + 1, stats);

        if (pc_val >= pc_beta) {
            pc_val = -alphaBeta(board, pc_depth, -pc_beta, -pc_beta + 1,
                                ply_from_root + 1, thread, tm, stats, false,
                                policy_advice.top_move, ss);
        }

        board.unmakeMove(policy_advice.top_move);
        thread.accumulatorStack.pop();

        if (pc_val >= pc_beta) {
            return pc_val;
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

    // ========================================================================
    // Move picker setup
    // ========================================================================

    chess::Move counter_move = g_counterMoves.get(previous_move);
    chess::Color side_to_move = board.sideToMove();

    MovePickerContext mpCtx(tt_move, counter_move, side_to_move, ply_from_root, ss);
    mpCtx.policy = use_policy_node ? &policy_advice : nullptr;

    MovePicker mp(board, mpCtx, depth, false, false);

    chess::Move best_move;
    int best_score = -MATE_SCORE;
    int original_alpha = alpha;

    chess::Move quiets_searched[MAX_QUIETS_TRACKED];
    int quiets_count = 0;

    CaptureSearchInfo captures_searched[MAX_CAPTURES_TRACKED];
    int captures_count = 0;

    int move_count = 0;
    bool had_non_excluded_move = false;

    // POLICY: Cumulative mass tracking
    float searched_policy_mass = 0.0f;
    int searched_quiets = 0;

    // POLICY: Precompute weights
    float protect_w = policyWeight(g_pt.policy_protect);
    float futility_w = policyWeight(g_pt.policy_futility);
    float see_w = policyWeight(g_pt.policy_see);
    float hist_prune_w = policyWeight(g_pt.policy_history_prune);
    float lmp_w = policyWeight(g_pt.policy_lmp);
    float whitelist_w = policyWeight(g_pt.policy_whitelist);
    float cum_w = policyWeight(g_pt.policy_cumulative_mass);
    float tail_w = policyWeight(g_pt.policy_tail_reduce);
    float top_lmr_w = policyWeight(g_pt.policy_top_protect_lmr);
    float int_lmr_w = policyWeight(g_pt.internal_policy_lmr);
    float ext_w = policyWeight(g_pt.policy_extension);
    float neg_ext_w = policyWeight(g_pt.policy_negative_extension);
    float hard_w = policyWeight(g_pt.policy_hard_prune);
    float se_gate_w = policyWeight(g_pt.policy_se_gate);

    while (true) {
        bool is_quiet = false;
        chess::Move move = mp.next(is_quiet);
        if (move == chess::Move()) break;
        if (move == excluded_move) continue;

        had_non_excluded_move = true;
        bool is_noisy = !is_quiet;

        // ====================================================================
        // POLICY: Compute protection flag
        // ====================================================================

        bool policy_protected = false;
        if (protect_w > 0.0f && use_policy_node && policy_advice.ok && is_quiet) {
            policy_protected = policy_advice.is_protected(move, protect_w);
        }

        // ====================================================================
        // POLICY: SEE threshold modulation
        // ====================================================================

        int see_threshold = -50 * depth;

        if (see_w > 0.0f && use_policy_node && policy_advice.ok && is_quiet) {
            int r = policy_advice.rank_of(move);
            if (r == 0) {
                see_threshold -= int(see_w * 100 * depth);
            } else if (r <= 2) {
                see_threshold -= int(see_w * 50 * depth);
            } else if (r >= 12) {
                see_threshold += int(see_w * 50 * depth);
            }
        }

        // Noisy SEE prune
        if (!in_singular_search && !is_pv_node && !in_check && is_noisy && depth <= 8 &&
            best_score > -MATE_SCORE + 100 &&
            !chess::see::see_ge(board, move, -50 * depth)) {
            continue;
        }

        // Quiet SEE prune (policy-modulated)
        if (!in_singular_search && !is_pv_node && !in_check && depth <= 8 &&
            is_quiet && move != tt_move && best_score > -MATE_SCORE + 100 &&
            !g_killerMoves.is_killer(ply_from_root, move) && move_count >= 2 &&
            !policy_protected) {
            if (!chess::see::see_ge(board, move, see_threshold)) {
                continue;
            }
        }

        // ====================================================================
        // POLICY: History pruning modulation
        // ====================================================================

        if (!in_singular_search && !is_pv_node && !in_check && depth <= 4 &&
            is_quiet && move_count >= 3 && move != tt_move &&
            best_score > -MATE_SCORE + 100 && !policy_protected) {
            chess::Piece hp = board.at(move.from());
            int hist_score = getCombinedHist(side_to_move, move, hp, ply_from_root, ss);

            int hist_threshold = -2000 * depth;

            if (hist_prune_w > 0.0f && use_policy_node && policy_advice.ok) {
                float rel = policy_advice.rel_of(move);
                if (rel < -1.5f) {
                    hist_threshold = int(hist_threshold * (1.0f + hist_prune_w));
                }
            }

            if (hist_score < hist_threshold) continue;
        }

        // ====================================================================
        // POLICY: LMP modulation
        // ====================================================================

        if (!in_singular_search && !is_pv_node && !in_check && depth <= 8 &&
            is_quiet && move != tt_move && best_score > -MATE_SCORE + 100 &&
            !policy_protected) {

            int lmp_limit = 3 + depth * depth;

            if (lmp_w > 0.0f && use_policy_node && policy_advice.ok) {
                int r = policy_advice.rank_of(move);
                if (r < 999) {
                    if (r <= 2) {
                        lmp_limit += int(lmp_w * 4);
                    } else if (r >= 10) {
                        lmp_limit -= int(lmp_w * 3);
                    }
                } else {
                    lmp_limit -= int(lmp_w * 2);
                }
                lmp_limit = std::max(2, lmp_limit);
            }

            if (move_count >= lmp_limit) break;
        }

        // ====================================================================
        // POLICY: Neural whitelist (NLMP)
        // ====================================================================

        if (whitelist_w > 0.0f && use_policy_node && policy_advice.ok &&
            !is_pv_node && !in_check && is_quiet &&
            move != tt_move &&
            !g_killerMoves.is_killer(ply_from_root, move) &&
            move != counter_move &&
            !policy_protected &&
            best_score > -MATE_SCORE + 100) {

            int r = policy_advice.rank_of(move);
            int base_whitelist = std::max(4, 3 + depth);
            int shrink = int(std::lround(4.0f * whitelist_w));
            int effective_whitelist = std::max(3, base_whitelist - shrink);

            if (r >= effective_whitelist && depth <= 8) {
                chess::Piece hp = board.at(move.from());
                int hist = getCombinedHist(side_to_move, move, hp, ply_from_root, ss);
                if (hist < 4000 && !chess::see::see_ge(board, move, 0)) {
                    continue;
                }
            }
        }

        // ====================================================================
        // POLICY: Hard prune (very high strength only)
        // ====================================================================

        if (hard_w > 0.70f && use_policy_node && policy_advice.ok &&
            !is_pv_node && !in_check && is_quiet &&
            depth <= 5 &&
            move != tt_move &&
            !g_killerMoves.is_killer(ply_from_root, move) &&
            move != counter_move &&
            !policy_protected &&
            best_score > -MATE_SCORE + 100) {

            int r = policy_advice.rank_of(move);
            float rel = policy_advice.rel_of(move);
            int whitelist = std::max(4, 3 + depth);

            if (r >= whitelist + 4 && rel < -2.5f) {
                chess::Piece hp = board.at(move.from());
                int hist = getCombinedHist(side_to_move, move, hp, ply_from_root, ss);
                if (hist < 0 && !chess::see::see_ge(board, move, 0)) {
                    continue;
                }
            }
        }

        // ====================================================================
        // Singular extensions (with policy SE gate)
        // ====================================================================

        int se_ext = 0;

        if (!in_singular_search && move == tt_move && !in_check) {
            // POLICY: SE gate — modify margin based on policy agreement
            int se_margin_adj = 0;
            if (se_gate_w > 0.0f && use_policy_node && policy_advice.ok) {
                int r = policy_advice.rank_of(tt_move);
                if (r == 0 && policy_advice.top_prob > 0.35f) {
                    se_margin_adj = int(se_gate_w * depth);
                } else if (r >= 10) {
                    se_margin_adj = -int(se_gate_w * depth);
                }
            }

            auto se = probeSingularExtension(board, depth, beta + se_margin_adj,
                                             ply_from_root, thread, tm, stats,
                                             tt_move, hash, is_pv_node, is_quiet, ss);
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

        // ====================================================================
        // POLICY: Futility margin modulation
        // ====================================================================

        if (!in_singular_search && !is_pv_node && !in_check && !gives_check &&
            depth <= 7 && is_quiet && move != tt_move &&
            best_score > -MATE_SCORE + 100 &&
            std::abs(alpha) < MATE_SCORE - 100 && !policy_protected) {

            int futility_margin = 100 + 80 * depth;

            if (futility_w > 0.0f && use_policy_node && policy_advice.ok) {
                int r = policy_advice.rank_of(move);
                if (r == 0) {
                    futility_margin += int(futility_w * 120 * depth);
                } else if (r <= 2) {
                    futility_margin += int(futility_w * 60 * depth);
                } else if (r >= 10) {
                    futility_margin -= int(futility_w * 80 * depth);
                }
            }

            if (static_eval + futility_margin <= alpha) {
                board.unmakeMove(move);
                thread.accumulatorStack.pop();
                continue;
            }
        }

        int eval;
        int local_extension = extension + se_ext;

        // ====================================================================
        // POLICY: Extension for confident top policy move
        // ====================================================================

        int policy_ext = 0;

        if (ext_w > 0.0f && use_policy_node && policy_advice.ok &&
            is_quiet && move == policy_advice.top_move &&
            policy_advice.sharpness > 0.45f &&
            std::abs(best_score) < MATE_SCORE - 500) {

            float required_prob = 0.50f - 0.20f * ext_w;
            if (policy_advice.top_prob > required_prob) {
                if (is_pv_node || depth <= 8) {
                    policy_ext = 1;
                }
            }
        }

        // ====================================================================
        // POLICY: Negative extension for policy trash
        // ====================================================================

        int policy_neg_ext = 0;

        if (neg_ext_w > 0.0f && use_policy_node && policy_advice.ok &&
            is_quiet && !policy_protected && depth <= 6 && !is_pv_node) {
            int r = policy_advice.rank_of(move);
            float rel = policy_advice.rel_of(move);
            if (r >= 12 && rel < -2.0f) {
                policy_neg_ext = -1;
            }
        }

        local_extension += policy_ext + policy_neg_ext;
        local_extension = std::clamp(local_extension, -1, 2);

        int new_depth = depth + local_extension - 1;

        bool can_reduce = !in_check && is_quiet && move_count > 1 &&
                          depth >= 3 && !gives_check && !in_singular_search &&
                          new_depth > 1;

        if (can_reduce) {
            int reduction = lmr_reductions[std::min(depth, 63)][std::min(move_count, 63)];

            if (move == tt_move) reduction = 0;
            else if (move_count <= 3) reduction = std::max(0, reduction - 1);

            if (!is_pv_node) reduction += 1;
            if (!improving) reduction += 1;

            int combined_hist = getCombinedHist(side_to_move, move, moved_piece,
                                                ply_from_root, ss);
            reduction -= std::clamp(combined_hist / 4096, -2, 2);

            // ================================================================
            // POLICY: Top protect LMR
            // ================================================================

            if (top_lmr_w > 0.0f && use_policy_node && policy_advice.ok && is_quiet) {
                int r = policy_advice.rank_of(move);
                if (r == 0 && policy_advice.sharpness > 0.30f) {
                    reduction -= int(std::lround(top_lmr_w * 2.0f));
                } else if (r <= 1 && policy_advice.sharpness > 0.40f) {
                    reduction -= int(std::lround(top_lmr_w * 1.0f));
                }
            }

            // ================================================================
            // POLICY: Internal policy LMR
            // ================================================================

            if (int_lmr_w > 0.0f && use_policy_node && policy_advice.ok && is_quiet) {
                float rel = policy_advice.rel_of(move);
                float sharp = policy_advice.sharpness;

                float adj = -0.80f * rel;
                adj = std::clamp(adj, -2.0f, 3.0f);
                adj *= sharp;
                adj *= int_lmr_w;

                reduction += int(std::lround(adj));
            }

            // ================================================================
            // POLICY: Tail reduce
            // ================================================================

            if (tail_w > 0.0f && use_policy_node && policy_advice.ok &&
                is_quiet && !policy_protected) {
                float rel = policy_advice.rel_of(move);
                if (rel < -1.0f) {
                    float extra = tail_w * std::clamp(-rel - 1.0f, 0.0f, 3.0f);
                    reduction += int(std::lround(extra));
                }
            }

            reduction = std::clamp(reduction, 0, new_depth - 1);

            if (reduction > 0) {
                eval = -alphaBeta(board, new_depth - reduction, -alpha - 1, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
                if (eval > alpha) {
                    eval = -alphaBeta(board, new_depth, -alpha - 1, -alpha,
                                      ply_from_root + 1, thread, tm, stats, true, move, ss);
                }
            } else {
                eval = -alphaBeta(board, new_depth, -alpha - 1, -alpha,
                                  ply_from_root + 1, thread, tm, stats, true, move, ss);
            }

            if (eval > alpha && eval < beta) {
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

        if (eval > best_score) { best_score = eval; best_move = move; }

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
                int bonus = 32 * depth * depth;
                CaptureSearchInfo ci;
                if (extractCaptureInfo(board, move, ci))
                    g_captureHistory.update(ci.piece_type, ci.to_sq, ci.captured_type, bonus);
            }
            {
                int malus = 32 * depth * depth;
                for (int c = 0; c < captures_count; ++c)
                    g_captureHistory.update(captures_searched[c].piece_type,
                                            captures_searched[c].to_sq,
                                            captures_searched[c].captured_type, -malus);
            }
            if (!in_singular_search)
                storeTT(hash, depth, beta, best_move, TT_LOWER, ply_from_root);
            return beta;
        }

        if (eval > alpha) alpha = eval;

        if (is_quiet && quiets_count < MAX_QUIETS_TRACKED)
            quiets_searched[quiets_count++] = move;

        if (is_noisy && captures_count < MAX_CAPTURES_TRACKED) {
            CaptureSearchInfo ci;
            if (extractCaptureInfo(board, move, ci))
                captures_searched[captures_count++] = ci;
        }

        // ====================================================================
        // POLICY: Cumulative mass break
        // ====================================================================

        if (is_quiet && use_policy_node && policy_advice.ok) {
            float p = policy_advice.prob_of(move);
            searched_policy_mass += p;
            searched_quiets++;
        }

        if (cum_w > 0.0f && use_policy_node && policy_advice.ok &&
            !is_pv_node && !in_check && searched_quiets >= 4) {
            int max_cum_depth = 5 + int(std::lround(3.0f * cum_w));
            if (depth <= max_cum_depth) {
                float required_mass = 0.995f - 0.06f * cum_w;
                if (searched_policy_mass > required_mass) {
                    break;
                }
            }
        }
    }

    if (!had_non_excluded_move) {
        if (in_check) return -MATE_SCORE + ply_from_root;
        return getDrawScore(ply_from_root);
    }

    // History updates for best move
    if (best_score > original_alpha && best_move != chess::Move()) {
        if (isQuietMove(board, best_move)) {
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
        } else {
            int bonus = 8 * depth * depth;
            CaptureSearchInfo ci;
            if (extractCaptureInfo(board, best_move, ci))
                g_captureHistory.update(ci.piece_type, ci.to_sq, ci.captured_type, bonus);
        }
        {
            int malus = 8 * depth * depth;
            for (int c = 0; c < captures_count; ++c) {
                if (captures_searched[c].move != best_move)
                    g_captureHistory.update(captures_searched[c].piece_type,
                                            captures_searched[c].to_sq,
                                            captures_searched[c].captured_type, -malus);
            }
        }
    }

    bool exact_node = best_score > original_alpha && best_score < beta;
    if (!in_singular_search && !in_check && exact_node &&
        best_move != chess::Move() && isQuietMove(board, best_move)) {
        updateCorrection(side_to_move, hash, depth, raw_static_eval, best_score);
    }

    if (!in_singular_search) {
        TTFlag flag = (best_score > original_alpha) ? TT_EXACT : TT_UPPER;
        storeTT(hash, depth, best_score, best_move, flag, ply_from_root);
    }

    return best_score;
}

// ============================================================================
// Root search with full policy integration
// ============================================================================

chess::Move search(chess::Board& board, int max_depth, ThreadInfo& thread, TimeManager& tm,
                   int64_t node_limit, int* score_out, uint64_t* nodes_out) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) return chess::Move();

    if (moves.size() == 1) {
        if (!g_silent) std::cout << "info string only move" << std::endl;
        if (score_out) *score_out = 0;
        return moves[0];
    }

    if (g_butterflyHistory.should_age()) {
        g_butterflyHistory.age();
        g_contHist1ply.age();
        g_contHist2ply.age();
        g_correctionHistory.age();
    }

    // Clear policy cache for this search
    g_policy_cache.clear();

    // ========================================================================
    // POLICY: Compute root policy once
    // ========================================================================

    RootPolicy rootPolicy;
    computeRootPolicy(board, rootPolicy);

    // Initial best move from policy
    chess::Move best_move;
    float root_init_w = policyWeight(g_pt.root_policy_order);
    if (root_init_w > 0.0f && rootPolicy.ok && rootPolicy.top_any != chess::Move()) {
        best_move = rootPolicy.top_any;
    } else {
        best_move = moves[0];
    }

    int best_score = -MATE_SCORE;
    double last_depth_ms = 100.0;
    const bool root_in_check = board.inCheck();

    // Policy TM state
    chess::Move prev_policy_top = chess::Move();
    int policy_stable_count = 0;

    SearchStack ss[MAX_PLY + 10];
    for (int i = 0; i < MAX_PLY + 10; ++i) {
        ss[i].static_eval = -MATE_SCORE;
        ss[i].current_move = chess::Move();
        ss[i].excluded_move = chess::Move();
        ss[i].moved_piece = chess::Piece::NONE;
    }

    SearchStats stats;
    stats.reset();

    if (node_limit > 0)
        tm.set_node_limit(node_limit, &stats.nodes);

    advanceTTGeneration();

    for (int depth = 1; depth <= max_depth; ++depth) {
        auto depth_start = std::chrono::high_resolution_clock::now();

        if (depth > 1 && !tm.should_continue_depth(depth, last_depth_ms))
            break;

        chess::Move depth_best_move = best_move;
        int score = -MATE_SCORE;

        // ====================================================================
        // POLICY: Aspiration entropy widening
        // ====================================================================

        int delta = 25;
        float asp_w = policyWeight(g_pt.aspiration_entropy);
        if (asp_w > 0.0f && rootPolicy.ok) {
            delta = int(delta * (1.0f + asp_w * rootPolicy.norm_entropy_any));
        }

        int aspiration_alpha, aspiration_beta;
        const bool root_lmr_enabled = depth >= 3 && moves.size() > 1;

        while (true) {
            if (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100) {
                aspiration_alpha = std::max(-MATE_SCORE, best_score - delta);
                aspiration_beta = std::min(MATE_SCORE, best_score + delta);
            } else {
                aspiration_alpha = -MATE_SCORE;
                aspiration_beta = MATE_SCORE;
            }

            int alpha = aspiration_alpha;
            int beta = aspiration_beta;
            score = -MATE_SCORE;
            depth_best_move = best_move;

            // ================================================================
            // POLICY: Root move ordering
            // ================================================================

            struct RootScoredMove {
                chess::Move move;
                int score;
                bool is_quiet;
            };

            std::vector<RootScoredMove> root_moves;
            root_moves.reserve(moves.size());

            float root_order_w = policyWeight(g_pt.root_policy_order);
            float root_tail_w = policyWeight(g_pt.root_policy_tail);
            float root_cand_w = policyWeight(g_pt.root_candidate);
            float sac_w = policyWeight(g_pt.policy_sacrifice);
            float bad_cap_w = policyWeight(g_pt.policy_bad_capture);

            for (const auto& move : moves) {
                bool is_quiet = isQuietMove(board, move);
                int s = 0;

                if (move == best_move && best_move != chess::Move()) {
                    s = 3000000;
                } else if (!is_quiet) {
                    // Capture scoring
                    chess::Piece attacker_piece = board.at(move.from());
                    chess::Piece captured_piece = chess::Piece::NONE;
                    if (move.typeOf() == chess::Move::ENPASSANT) {
                        chess::Square capSq(move.to().file(), move.from().rank());
                        captured_piece = board.at(capSq);
                    } else {
                        captured_piece = board.at(move.to());
                    }
                    int victim = (captured_piece != chess::Piece::NONE)
                                 ? pieceValue(captured_piece.type()) : 0;
                    int attacker = pieceValue(attacker_piece.type());
                    int tactical = victim * 10 - attacker;
                    if (move.typeOf() == chess::Move::PROMOTION)
                        tactical += 600000 + pieceValue(move.promotionType());
                    if (captured_piece != chess::Piece::NONE)
                        tactical += g_captureHistory.get(
                            static_cast<int>(attacker_piece.type()),
                            move.to().index(),
                            static_cast<int>(captured_piece.type())) / 16;

                    bool good_see = chess::see::see_ge(board, move, 0);

                    if (good_see) {
                        s = 2000000 + tactical;
                    } else {
                        s = -1000000 + tactical;

                        // PolicySacrifice: promote policy-loved bad captures
                        if (sac_w > 0.0f && rootPolicy.ok) {
                            int idx = rootPolicy.find(move);
                            if (idx >= 0 && rootPolicy.quiet_rank[idx] >= 0 &&
                                rootPolicy.quiet_rank[idx] < 3) {
                                s += int(sac_w * 1900000.0f);
                            }
                        }

                        // PolicyBadCapture: suppress policy-hated bad captures
                        if (bad_cap_w > 0.0f && rootPolicy.ok) {
                            int idx = rootPolicy.find(move);
                            if (idx >= 0 && rootPolicy.quiet_rank[idx] >= 12) {
                                s -= int(bad_cap_w * 500000.0f);
                            }
                        }
                    }
                } else {
                    // Quiet scoring
                    s = g_butterflyHistory.get(board.sideToMove(), move.from(), move.to());

                    if (g_killerMoves.is_killer(0, move))
                        s += 1500000;

                    // RootPolicyOrder: policy ordering bonus
                    if (root_order_w > 0.0f && rootPolicy.ok) {
                        int idx = rootPolicy.find(move);
                        if (idx >= 0 && rootPolicy.quiet_rank[idx] >= 0) {
                            float rel = rootPolicy.rel[idx];
                            float sharp = rootPolicy.quiet_sharpness;

                            int policy_bonus = int(root_order_w * 1600.0f * rel * sharp);
                            policy_bonus = std::clamp(policy_bonus, -5000, 8000);

                            int r = rootPolicy.quiet_rank[idx];
                            int rank_bonus = 0;
                            if (r == 0) rank_bonus = 6000;
                            else if (r == 1) rank_bonus = 3500;
                            else if (r <= 2) rank_bonus = 2000;
                            else if (r <= 5) rank_bonus = 900;
                            else if (r <= 10) rank_bonus = 250;

                            policy_bonus += int(rank_bonus * sharp * root_order_w);
                            s += policy_bonus;
                        }
                    }
                }

                root_moves.push_back({move, s, is_quiet});
            }

            std::stable_sort(root_moves.begin(), root_moves.end(),
                             [](const RootScoredMove& a, const RootScoredMove& b) {
                                 return a.score > b.score;
                             });

            // ================================================================
            // Root move loop
            // ================================================================

            int root_move_count = 0;

            for (size_t rmi = 0; rmi < root_moves.size(); ++rmi) {
                chess::Move move = root_moves[rmi].move;
                bool root_is_quiet = root_moves[rmi].is_quiet;

                if (tm.should_stop()) goto search_done;

                chess::Piece root_piece = board.at(move.from());

                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);

                ss[0].current_move = move;
                ss[0].moved_piece = root_piece;

                bool gives_check = board.inCheck();
                int eval;

                bool is_draw_move = isDrawByRepetition(board) || isDrawByFiftyMove(board);

                if (is_draw_move) {
                    eval = -getDrawScore(1);
                } else if (root_move_count == 0) {
                    eval = -alphaBeta(board, depth - 1, -beta, -alpha, 1,
                                      thread, &tm, stats, true, move, ss);
                } else {
                    int new_depth = depth - 1;
                    int reduction = 0;

                    bool can_reduce_root =
                        root_lmr_enabled &&
                        !root_in_check &&
                        !gives_check &&
                        root_is_quiet &&
                        move != best_move &&
                        new_depth > 1;

                    if (can_reduce_root) {
                        int move_no = root_move_count + 1;
                        reduction = lmr_reductions[std::min(depth, 63)]
                                                [std::min(move_no, 63)];

                        // ====================================================
                        // POLICY: Root LMR modulation
                        // ====================================================

                        float root_lmr_w = policyWeight(g_pt.root_policy_lmr);

                        if (root_lmr_w > 0.0f && rootPolicy.ok && root_is_quiet) {
                            int idx = rootPolicy.find(move);
                            if (idx >= 0 && rootPolicy.quiet_rank[idx] >= 0) {
                                float rel = rootPolicy.rel[idx];
                                float sharp = rootPolicy.quiet_sharpness;

                                float adj = -0.85f * rel;
                                adj = std::clamp(adj, -2.0f, 3.0f);
                                adj *= sharp;
                                adj *= root_lmr_w;

                                reduction += int(std::lround(adj));
                            }
                        }

                        // ====================================================
                        // POLICY: Root tail reduction
                        // ====================================================

                        if (root_tail_w > 0.0f && rootPolicy.ok && root_is_quiet) {
                            float rel = rootPolicy.rel_of(move);
                            if (rel < -1.0f && rel > -999.0f) {
                                float extra = root_tail_w * std::clamp(-rel - 1.0f, 0.0f, 3.0f);
                                reduction += int(std::lround(extra));
                            }
                        }

                        // ====================================================
                        // POLICY: Root candidate classes
                        // ====================================================

                        if (root_cand_w > 0.0f && rootPolicy.ok && root_is_quiet) {
                            int idx = rootPolicy.find(move);
                            if (idx >= 0 && rootPolicy.quiet_rank[idx] >= 0) {
                                int r = rootPolicy.quiet_rank[idx];
                                if (r <= 2) {
                                    reduction -= int(std::lround(root_cand_w * 1.0f));
                                } else if (r >= 10) {
                                    reduction += int(std::lround(root_cand_w * 1.0f));
                                }
                            }
                        }

                        reduction = std::clamp(reduction, 0, std::max(0, new_depth - 1));
                    }

                    eval = -alphaBeta(board, new_depth - reduction, -alpha - 1, -alpha, 1,
                                      thread, &tm, stats, true, move, ss);

                    // ========================================================
                    // POLICY: Root verification margin
                    // ========================================================

                    if (reduction > 0) {
                        float verify_w = policyWeight(g_pt.root_policy_verify);
                        int verify_margin = 0;

                        if (verify_w > 0.0f && rootPolicy.ok && root_is_quiet) {
                            if (rootPolicy.protected_quiet(move, verify_w)) {
                                verify_margin = 20;
                            }
                        }

                        if (eval > alpha - verify_margin) {
                            eval = -alphaBeta(board, new_depth, -alpha - 1, -alpha, 1,
                                              thread, &tm, stats, true, move, ss);
                        }
                    }

                    if (eval > alpha && eval < beta) {
                        eval = -alphaBeta(board, new_depth, -beta, -alpha, 1,
                                          thread, &tm, stats, true, move, ss);
                    }
                }

                board.unmakeMove(move);
                thread.accumulatorStack.pop();

                // ============================================================
                // POLICY: Root tie-breaking
                // ============================================================

                float tie_w = policyWeight(g_pt.root_tie_break);
                bool update_best = (eval > score);

                if (!update_best && tie_w > 0.0f && rootPolicy.ok &&
                    std::abs(eval - score) <= 3 && root_is_quiet) {
                    float p_move = rootPolicy.prob_any_of(move);
                    float p_current = rootPolicy.prob_any_of(depth_best_move);
                    if (p_move > p_current) {
                        update_best = true;
                    }
                }

                if (update_best) {
                    score = eval;
                    depth_best_move = move;
                }

                if (eval > alpha) alpha = eval;
                if (eval >= beta) break;

                root_move_count++;
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

        storeTT(getZobristHash(board), depth, best_score, best_move, TT_EXACT, 0, true);

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

        if (!g_silent)
            std::cout << "info score " << score_str
                      << " depth " << depth
                      << " nodes " << stats.nodes
                      << " nps " << nps << " time "
                      << elapsed << " pv " << pv_str << "\n";

        tm.update_stability(best_move);

        // ====================================================================
        // POLICY: Time management
        // ====================================================================

        if (rootPolicy.ok &&
            depth >= 6 &&
            node_limit <= 0 &&
            tm.soft_limit_ms < tm.hard_limit_ms) {

            double scale = 1.0;

            // PolicyTMContinuous: ratio-based scaling
            float tm_cont_w = policyWeight(g_pt.policy_tm_continuous);
            if (tm_cont_w > 0.0f) {
                float p_top = rootPolicy.top_prob_any;
                float p_search = rootPolicy.prob_any_of(best_move);
                float ratio = p_search / std::max(1e-6f, p_top);

                double raw_scale;
                if (ratio >= 0.80f && p_top >= 0.30f) {
                    raw_scale = 0.85;
                } else if (ratio >= 0.50f) {
                    raw_scale = 1.00;
                } else if (ratio >= 0.20f) {
                    raw_scale = 1.15;
                } else {
                    raw_scale = 1.35;
                }

                scale = 1.0 + tm_cont_w * (raw_scale - 1.0);
            }

            // PolicyEntropyTM: entropy-based scaling
            float ent_w = policyWeight(g_pt.policy_entropy_tm);
            if (ent_w > 0.0f) {
                if (rootPolicy.norm_entropy_any > 0.85f) {
                    scale *= (1.0 + ent_w * 0.15);
                } else if (rootPolicy.norm_entropy_any < 0.45f) {
                    scale *= (1.0 - ent_w * 0.10);
                }
            }

            // PolicyStabilityTM: stability-based scaling
            float stab_w = policyWeight(g_pt.policy_stability_tm);
            if (stab_w > 0.0f) {
                chess::Move cur_policy_top = rootPolicy.top_any;
                if (cur_policy_top == prev_policy_top) {
                    policy_stable_count++;
                } else {
                    policy_stable_count = 0;
                }
                prev_policy_top = cur_policy_top;

                if (policy_stable_count >= 3) {
                    scale *= (1.0 - stab_w * 0.10);
                } else if (policy_stable_count == 0 && depth > 8) {
                    scale *= (1.0 + stab_w * 0.10);
                }
            }

            // PolicyDisagreeTime: disagreement-based scaling
            float disagree_w = policyWeight(g_pt.policy_disagree_time);
            if (disagree_w > 0.0f) {
                if (rootPolicy.top_any != best_move) {
                    float p_top = rootPolicy.top_prob_any;
                    float p_search = rootPolicy.prob_any_of(best_move);
                    float gap = p_top - p_search;

                    if (gap > 0.25f) {
                        scale *= (1.0 + disagree_w * 0.30);
                    } else if (gap > 0.10f) {
                        scale *= (1.0 + disagree_w * 0.15);
                    }
                }
            }

            scale = std::clamp(scale, 0.70, 1.60);
            tm.set_policy_time_scale(scale);

            if (!g_silent) {
                std::cout << "info string policy_tm"
                          << " depth " << depth
                          << " scale " << scale
                          << " pol " << chess::uci::moveToUci(rootPolicy.top_any)
                          << " (" << (rootPolicy.top_prob_any * 100.f) << "%)"
                          << " search " << chess::uci::moveToUci(best_move)
                          << " ent " << rootPolicy.norm_entropy_any
                          << std::endl;
            }
        } else {
            tm.set_policy_time_scale(1.0);
        }
    }

search_done:

    if (score_out) *score_out = best_score;
    if (nodes_out) *nodes_out = stats.nodes;

    std::cerr << "info string total nodes: " << stats.nodes << std::endl;

    return best_move;
}