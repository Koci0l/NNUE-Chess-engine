// ============================================================================
// movepick.cpp
// ============================================================================
#include "movepick.h"
#include "history.h"
#include "see.h"
#include "policy.h"
#include <algorithm>
#include <cstring>

// ============================================================================
// MovePicker
// ============================================================================

MovePicker::MovePicker(const chess::Board& board, const MovePickerContext& ctx,
                       int depth, bool skip_quiets, bool use_policy)
    : m_board(board), m_ctx(ctx), m_depth(depth),
      m_skip_quiets(skip_quiets), m_use_policy(use_policy),
      m_stage(MovePickStage::TT_MOVE),
      m_last_score(0), m_last_policy_rank(-1),
      m_policy_ready(false), m_policy_nq(0), m_policy_sharp(0.0f)
{
    m_killer1 = g_killerMoves.get(ctx.ply, 0);
    m_killer2 = g_killerMoves.get(ctx.ply, 1);
    for (int i = 0; i < 256; ++i) m_policy_rank[i] = -1;
}

// ============================================================================
// Legality generation (once)
// ============================================================================

void MovePicker::ensureLegal() {
    if (m_legal_generated) return;
    chess::movegen::legalmoves(m_all_legal, m_board);
    m_legal_generated = true;
}

bool MovePicker::isValid(const chess::Move& move) const {
    for (const auto& m : m_all_legal) {
        if (m == move) return true;
    }
    return false;
}

// ============================================================================
// Scoring helpers
// ============================================================================

int MovePicker::scoreOneCapture(const chess::Move& move) {
    chess::Piece attacker_piece = m_board.at(move.from());
    chess::Piece captured_piece = chess::Piece::NONE;

    if (move.typeOf() == chess::Move::ENPASSANT) {
        chess::Square capSq(move.to().file(), move.from().rank());
        captured_piece = m_board.at(capSq);
    } else {
        captured_piece = m_board.at(move.to());
    }

    int victim = 0;
    if (captured_piece != chess::Piece::NONE) {
        victim = pieceValue(captured_piece.type());
    }
    int attacker = pieceValue(attacker_piece.type());

    int score = victim * 10 - attacker;

    if (move.typeOf() == chess::Move::PROMOTION) {
        score += 600000 + pieceValue(move.promotionType());
    }

    // Capture history
    if (captured_piece != chess::Piece::NONE) {
        score += g_captureHistory.get(
            static_cast<int>(attacker_piece.type()),
            move.to().index(),
            static_cast<int>(captured_piece.type())) / 16;
    }

    return score;
}

int MovePicker::scoreOneQuiet(const chess::Move& move) {
    chess::Color side = m_ctx.side_to_move;
    chess::Piece piece = m_board.at(move.from());

    int score = 0;

    // Butterfly history
    score += g_butterflyHistory.get(side, move.from(), move.to());

    // Continuation histories
    if (m_ctx.ply >= 1 && m_ctx.ss &&
        m_ctx.ss[m_ctx.ply - 1].moved_piece != chess::Piece::NONE) {
        score += g_contHist1ply.get(
            m_ctx.ss[m_ctx.ply - 1].moved_piece,
            m_ctx.ss[m_ctx.ply - 1].current_move.to(),
            piece, move.to());
    }
    if (m_ctx.ply >= 2 && m_ctx.ss &&
        m_ctx.ss[m_ctx.ply - 2].moved_piece != chess::Piece::NONE) {
        score += g_contHist2ply.get(
            m_ctx.ss[m_ctx.ply - 2].moved_piece,
            m_ctx.ss[m_ctx.ply - 2].current_move.to(),
            piece, move.to());
    }

    return score;
}

// ============================================================================
// Stage scoring
// ============================================================================

void MovePicker::scoreCaptures() {
    ensureLegal();
    m_capture_count = 0;
    m_bad_capture_count = 0;

    for (const auto& move : m_all_legal) {
        if (move == m_ctx.tt_move) continue;

        bool is_capture = m_board.at(move.to()) != chess::Piece::NONE ||
                          move.typeOf() == chess::Move::ENPASSANT;
        bool is_promo = move.typeOf() == chess::Move::PROMOTION;
        if (!is_capture && !is_promo) continue;

        int score = scoreOneCapture(move);

        if (chess::see::see_ge(m_board, move, 0)) {
            if (m_capture_count < 256) {
                m_captures[m_capture_count].move = move;
                m_captures[m_capture_count].score = score;
                ++m_capture_count;
            }
        } else {
            if (m_bad_capture_count < 256) {
                m_bad_captures[m_bad_capture_count].move = move;
                m_bad_captures[m_bad_capture_count].score = score;
                ++m_bad_capture_count;
            }
        }
    }
}

// ============================================================================
// Lazy policy computation — only called from scoreQuiets()
// ============================================================================

void MovePicker::computePolicyRanks() {
    m_policy_ready = false;
    m_policy_nq = 0;
    m_policy_sharp = 0.0f;

    if (!m_use_policy || !g_policy.loaded) return;
    if (m_all_legal.size() == 0 || m_all_legal.size() > 256) return;

    if (!g_policy.rankLegalQuiets(m_board, m_all_legal, m_policy_rank, &m_policy_nq))
        return;

    if (m_policy_nq < 4) return;

    // Sharpness proxy from quiet count (fewer quiets = more forced).
    // Replace with actual entropy if you modify rankLegalQuiets to output logits.
    if (m_policy_nq <= 6)       m_policy_sharp = 0.85f;
    else if (m_policy_nq <= 12) m_policy_sharp = 0.60f;
    else if (m_policy_nq <= 20) m_policy_sharp = 0.40f;
    else                        m_policy_sharp = 0.20f;

    m_policy_ready = true;
}

// ============================================================================
// scoreQuiets — with policy blending
// ============================================================================

void MovePicker::scoreQuiets() {
    ensureLegal();
    m_quiet_count = 0;
    m_quiet_idx = 0;

    // Lazy policy: computed here, only when we actually reach quiet generation.
    // Nodes that cut off on TT move / captures / killers never call this.
    computePolicyRanks();

    for (int idx = 0; idx < (int)m_all_legal.size(); ++idx) {
        const chess::Move move = m_all_legal[idx];

        if (move == m_ctx.tt_move) continue;

        bool is_capture = m_board.at(move.to()) != chess::Piece::NONE ||
                          move.typeOf() == chess::Move::ENPASSANT;
        bool is_promo = move.typeOf() == chess::Move::PROMOTION;
        if (is_capture || is_promo) continue;

        if (move == m_killer1 || move == m_killer2 ||
            move == m_ctx.counter_move) continue;

        if (m_quiet_count >= 256) break;

        int score = scoreOneQuiet(move);

        // Blend policy rank bonus
        if (m_policy_ready && m_policy_rank[idx] >= 0) {
            const int r = m_policy_rank[idx];
            int bonus = 0;
            if      (r == 0)  bonus = 8000;
            else if (r == 1)  bonus = 5000;
            else if (r == 2)  bonus = 3000;
            else if (r <= 5)  bonus = 1200;
            else if (r <= 10) bonus = 300;
            else if (r >= 20) bonus = -1000;

            score += (int)(bonus * m_policy_sharp);
        }

        m_quiets[m_quiet_count].move = move;
        m_quiets[m_quiet_count].score = score;
        ++m_quiet_count;
    }
}

// ============================================================================
// next() — staged iteration
// ============================================================================

chess::Move MovePicker::next(bool& is_quiet_out) {
    is_quiet_out = false;
    m_last_policy_rank = -1;

    for (;;) {
        switch (m_stage) {

        case MovePickStage::TT_MOVE: {
            m_stage = MovePickStage::GENERATE_CAPTURES;
            if (m_ctx.tt_move != chess::Move() && isValid(m_ctx.tt_move)) {
                m_last_score = 3000000;
                is_quiet_out = policyQuietLocal(m_board, m_ctx.tt_move);
                return m_ctx.tt_move;
            }
            break;
        }

        case MovePickStage::GENERATE_CAPTURES: {
            scoreCaptures();
            m_capture_idx = 0;
            m_stage = MovePickStage::GOOD_CAPTURES;
            break;
        }

        case MovePickStage::GOOD_CAPTURES: {
            if (m_capture_idx >= m_capture_count) {
                m_stage = MovePickStage::KILLER_1;
                break;
            }
            // Partial selection sort
            int best_i = m_capture_idx;
            for (int i = m_capture_idx + 1; i < m_capture_count; ++i) {
                if (m_captures[i].score > m_captures[best_i].score)
                    best_i = i;
            }
            std::swap(m_captures[m_capture_idx], m_captures[best_i]);
            const chess::Move mv = m_captures[m_capture_idx].move;
            m_last_score = m_captures[m_capture_idx].score;
            ++m_capture_idx;
            is_quiet_out = false;
            return mv;
        }

        case MovePickStage::KILLER_1: {
            m_stage = MovePickStage::KILLER_2;
            if (m_killer1 != chess::Move() && m_killer1 != m_ctx.tt_move &&
                isValid(m_killer1) && policyQuietLocal(m_board, m_killer1)) {
                m_last_score = 1500000;
                is_quiet_out = true;
                return m_killer1;
            }
            break;
        }

        case MovePickStage::KILLER_2: {
            m_stage = MovePickStage::COUNTER_MOVE;
            if (m_killer2 != chess::Move() && m_killer2 != m_ctx.tt_move &&
                m_killer2 != m_killer1 && isValid(m_killer2) &&
                policyQuietLocal(m_board, m_killer2)) {
                m_last_score = 1400000;
                is_quiet_out = true;
                return m_killer2;
            }
            break;
        }

        case MovePickStage::COUNTER_MOVE: {
            m_stage = MovePickStage::GENERATE_QUIETS;
            if (m_ctx.counter_move != chess::Move() &&
                m_ctx.counter_move != m_ctx.tt_move &&
                m_ctx.counter_move != m_killer1 &&
                m_ctx.counter_move != m_killer2 &&
                isValid(m_ctx.counter_move) &&
                policyQuietLocal(m_board, m_ctx.counter_move)) {
                m_last_score = 1300000;
                is_quiet_out = true;
                return m_ctx.counter_move;
            }
            break;
        }

        case MovePickStage::GENERATE_QUIETS: {
            if (m_skip_quiets) {
                m_stage = MovePickStage::BAD_CAPTURES;
                break;
            }
            scoreQuiets();
            m_stage = MovePickStage::QUIETS;
            break;
        }

        case MovePickStage::QUIETS: {
            if (m_quiet_idx >= m_quiet_count) {
                m_stage = MovePickStage::BAD_CAPTURES;
                break;
            }
            // Partial selection sort
            int best_i = m_quiet_idx;
            for (int i = m_quiet_idx + 1; i < m_quiet_count; ++i) {
                if (m_quiets[i].score > m_quiets[best_i].score)
                    best_i = i;
            }
            std::swap(m_quiets[m_quiet_idx], m_quiets[best_i]);

            const chess::Move mv = m_quiets[m_quiet_idx].move;
            m_last_score = m_quiets[m_quiet_idx].score;
            ++m_quiet_idx;

            // Expose policy rank for LMR in search.cpp
            if (m_policy_ready) {
                for (int i = 0; i < (int)m_all_legal.size(); ++i) {
                    if (m_all_legal[i] == mv) {
                        m_last_policy_rank = m_policy_rank[i];
                        break;
                    }
                }
            }

            is_quiet_out = true;
            return mv;
        }

        case MovePickStage::BAD_CAPTURES: {
            if (m_bad_capture_idx >= m_bad_capture_count) {
                m_stage = MovePickStage::DONE;
                break;
            }
            const chess::Move mv = m_bad_captures[m_bad_capture_idx].move;
            m_last_score = m_bad_captures[m_bad_capture_idx].score;
            ++m_bad_capture_idx;
            is_quiet_out = false;
            return mv;
        }

        case MovePickStage::DONE: {
            return chess::Move();
        }

        } // switch
    } // for
}

// ============================================================================
// QSearchMovePicker
// ============================================================================

QSearchMovePicker::QSearchMovePicker(const chess::Board& board, chess::Move tt_move,
                                     bool in_check)
    : m_board(board), m_tt_move(tt_move), m_in_check(in_check),
      m_stage(QMovePickStage::TT_MOVE),
      m_last_score(0)
{
}

void QSearchMovePicker::ensureLegal() {
    if (m_legal_generated) return;
    chess::movegen::legalmoves(m_legal, m_board);
    m_legal_generated = true;
}

bool QSearchMovePicker::isValid(const chess::Move& move) const {
    for (const auto& m : m_legal) {
        if (m == move) return true;
    }
    return false;
}

void QSearchMovePicker::pickBest(ScoredMove* moves, int start, int end) {
    int best_i = start;
    for (int i = start + 1; i < end; ++i) {
        if (moves[i].score > moves[best_i].score)
            best_i = i;
    }
    std::swap(moves[start], moves[best_i]);
}

void QSearchMovePicker::scoreCaptures() {
    ensureLegal();
    m_move_count = 0;

    for (const auto& move : m_legal) {
        if (move == m_tt_move) continue;

        bool is_capture = m_board.at(move.to()) != chess::Piece::NONE ||
                          move.typeOf() == chess::Move::ENPASSANT;
        bool is_promo = move.typeOf() == chess::Move::PROMOTION;

        if (!m_in_check && !is_capture && !is_promo) continue;

        if (m_move_count >= 256) break;

        int score = 0;
        if (is_capture || is_promo) {
            chess::Piece attacker_piece = m_board.at(move.from());
            chess::Piece captured_piece = chess::Piece::NONE;
            if (move.typeOf() == chess::Move::ENPASSANT) {
                chess::Square capSq(move.to().file(), move.from().rank());
                captured_piece = m_board.at(capSq);
            } else {
                captured_piece = m_board.at(move.to());
            }
            int victim = (captured_piece != chess::Piece::NONE)
                             ? pieceValue(captured_piece.type()) : 0;
            int attacker = pieceValue(attacker_piece.type());
            score = victim * 10 - attacker;
            if (is_promo) score += 600000 + pieceValue(move.promotionType());
        }

        m_moves[m_move_count].move = move;
        m_moves[m_move_count].score = score;
        ++m_move_count;
    }
}

chess::Move QSearchMovePicker::next() {
    for (;;) {
        switch (m_stage) {

        case QMovePickStage::TT_MOVE: {
            m_stage = QMovePickStage::GENERATE_CAPTURES;
            if (m_tt_move != chess::Move() && isValid(m_tt_move)) {
                m_last_score = 3000000;
                return m_tt_move;
            }
            break;
        }

        case QMovePickStage::GENERATE_CAPTURES: {
            scoreCaptures();
            m_move_idx = 0;
            m_stage = QMovePickStage::CAPTURES;
            break;
        }

        case QMovePickStage::CAPTURES: {
            if (m_move_idx >= m_move_count) {
                m_stage = QMovePickStage::DONE;
                break;
            }
            pickBest(m_moves, m_move_idx, m_move_count);
            const chess::Move mv = m_moves[m_move_idx].move;
            m_last_score = m_moves[m_move_idx].score;
            ++m_move_idx;
            return mv;
        }

        case QMovePickStage::DONE: {
            return chess::Move();
        }

        } // switch
    } // for
}

// ============================================================================
// Free functions (used by root scoring / diagnostics)
// ============================================================================

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move,
                         const MovePickerContext& ctx) {
    bool is_capture = board.at(move.to()) != chess::Piece::NONE ||
                      move.typeOf() == chess::Move::ENPASSANT;
    bool is_promo = move.typeOf() == chess::Move::PROMOTION;

    if (is_capture || is_promo) {
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
        int score = 2000000 + victim * 10 - attacker;
        if (is_promo) score += 600000 + pieceValue(move.promotionType());
        return score;
    }

    int score = 0;
    if (g_killerMoves.is_killer(ctx.ply, move)) score += 1500000;
    score += g_butterflyHistory.get(ctx.side_to_move, move.from(), move.to());
    return score;
}

std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves,
                                   const chess::Board& board,
                                   const MovePickerContext& ctx) {
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());
    for (const auto& move : moves) {
        ScoredMove sm;
        sm.move = move;
        sm.score = scoreMoveForOrdering(board, move, ctx);
        scored.push_back(sm);
    }
    return scored;
}

void pickNextMove(std::vector<ScoredMove>& moves, size_t current) {
    if (current >= moves.size()) return;
    size_t best_i = current;
    for (size_t i = current + 1; i < moves.size(); ++i) {
        if (moves[i].score > moves[best_i].score)
            best_i = i;
    }
    std::swap(moves[current], moves[best_i]);
}