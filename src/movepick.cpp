#include "movepick.h"
#include "history.h"
#include "see.h"

#include <algorithm>

// ============================================================================
// MovePicker
// ============================================================================

MovePicker::MovePicker(const chess::Board& board, const MovePickerContext& ctx,
                       int depth, bool skip_quiets)
    : m_board(board), m_ctx(ctx), m_depth(depth), m_skip_quiets(skip_quiets),
      m_stage(MovePickStage::TT_MOVE),
      m_capture_count(0), m_capture_idx(0),
      m_bad_capture_count(0), m_bad_capture_idx(0),
      m_quiet_count(0), m_quiet_idx(0),
      m_last_score(0), m_returned_count(0),
      m_legal_generated(false) {
    m_killer1 = g_killerMoves.get_killer(ctx.ply, 0);
    m_killer2 = g_killerMoves.get_killer(ctx.ply, 1);
}

bool MovePicker::wasReturned(const chess::Move& move) const {
    for (int i = 0; i < m_returned_count; ++i) {
        if (m_returned[i] == move) return true;
    }
    return false;
}

void MovePicker::markReturned(const chess::Move& move) {
    if (m_returned_count < 512) {
        m_returned[m_returned_count++] = move;
    }
}

void MovePicker::ensureLegal() {
    if (!m_legal_generated) {
        chess::movegen::legalmoves(m_all_legal, m_board);
        m_legal_generated = true;
    }
}

bool MovePicker::isValid(const chess::Move& move) const {
    if (move == chess::Move()) return false;
    for (const auto& m : m_all_legal) {
        if (m == move) return true;
    }
    return false;
}

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

    if (captured_piece != chess::Piece::NONE) {
        int cap_hist = g_captureHistory.get(
            static_cast<int>(attacker_piece.type()),
            move.to().index(),
            static_cast<int>(captured_piece.type()));
        score += cap_hist / 16;
    }

    return score;
}

int MovePicker::scoreOneQuiet(const chess::Move& move) {
    chess::Piece piece = m_board.at(move.from());
    int hist = g_butterflyHistory.get(m_ctx.side_to_move, move.from(), move.to());

    int cont1 = 0;
    int cont2 = 0;

    if (m_ctx.ss != nullptr) {
        if (m_ctx.ply >= 1 &&
            m_ctx.ss[m_ctx.ply - 1].moved_piece != chess::Piece::NONE) {
            cont1 = g_contHist1ply.get(
                m_ctx.ss[m_ctx.ply - 1].moved_piece,
                m_ctx.ss[m_ctx.ply - 1].current_move.to(),
                piece,
                move.to());
        }

        if (m_ctx.ply >= 2 &&
            m_ctx.ss[m_ctx.ply - 2].moved_piece != chess::Piece::NONE) {
            cont2 = g_contHist2ply.get(
                m_ctx.ss[m_ctx.ply - 2].moved_piece,
                m_ctx.ss[m_ctx.ply - 2].current_move.to(),
                piece,
                move.to());
        }
    }

    return hist + cont1 + cont2;
}

void MovePicker::scoreCaptures() {
    ensureLegal();

    m_capture_count = 0;
    m_capture_idx = 0;
    m_bad_capture_count = 0;
    m_bad_capture_idx = 0;

    for (const auto& move : m_all_legal) {
        if (move == m_ctx.tt_move) continue;

        bool is_capture = m_board.at(move.to()) != chess::Piece::NONE ||
                          move.typeOf() == chess::Move::ENPASSANT;
        bool is_promo = move.typeOf() == chess::Move::PROMOTION;

        if (!is_capture && !is_promo) continue;
        if (m_capture_count >= 256) break;

        m_captures[m_capture_count].move = move;
        m_captures[m_capture_count].score = scoreOneCapture(move);
        ++m_capture_count;
    }
}

void MovePicker::scoreQuiets() {
    ensureLegal();

    m_quiet_count = 0;
    m_quiet_idx = 0;

    for (const auto& move : m_all_legal) {
        if (move == m_ctx.tt_move) continue;

        bool is_capture = m_board.at(move.to()) != chess::Piece::NONE ||
                          move.typeOf() == chess::Move::ENPASSANT;
        bool is_promo = move.typeOf() == chess::Move::PROMOTION;

        if (is_capture || is_promo) continue;
        if (move == m_killer1 || move == m_killer2 || move == m_ctx.counter_move) continue;
        if (m_quiet_count >= 256) break;

        m_quiets[m_quiet_count].move = move;
        m_quiets[m_quiet_count].score = scoreOneQuiet(move);
        ++m_quiet_count;
    }
}

chess::Move MovePicker::next(bool& is_quiet_out) {
    is_quiet_out = false;

    while (true) {
        switch (m_stage) {
            case MovePickStage::TT_MOVE: {
                m_stage = MovePickStage::GENERATE_CAPTURES;

                if (m_ctx.tt_move != chess::Move()) {
                    ensureLegal();
                    if (isValid(m_ctx.tt_move)) {
                        m_last_score = 3000000;

                        bool tt_capture = m_board.at(m_ctx.tt_move.to()) != chess::Piece::NONE ||
                                          m_ctx.tt_move.typeOf() == chess::Move::ENPASSANT ||
                                          m_ctx.tt_move.typeOf() == chess::Move::PROMOTION;
                        is_quiet_out = !tt_capture;
                        return m_ctx.tt_move;
                    }
                }
                break;
            }

            case MovePickStage::GENERATE_CAPTURES: {
                scoreCaptures();
                m_stage = MovePickStage::GOOD_CAPTURES;
                break;
            }

            case MovePickStage::GOOD_CAPTURES: {
                while (m_capture_idx < m_capture_count) {
                    int best = m_capture_idx;
                    for (int j = m_capture_idx + 1; j < m_capture_count; ++j) {
                        if (m_captures[j].score > m_captures[best].score) {
                            best = j;
                        }
                    }
                    if (best != m_capture_idx) {
                        std::swap(m_captures[m_capture_idx], m_captures[best]);
                    }

                    chess::Move move = m_captures[m_capture_idx].move;
                    int score = m_captures[m_capture_idx].score;
                    ++m_capture_idx;

                    if (!chess::see::see_ge(m_board, move, 0)) {
                        if (m_bad_capture_count < 256) {
                            m_bad_captures[m_bad_capture_count].move = move;
                            m_bad_captures[m_bad_capture_count].score = score;
                            ++m_bad_capture_count;
                        }
                        continue;
                    }

                    m_last_score = 2000000 + score;
                    is_quiet_out = false;
                    return move;
                }

                m_stage = MovePickStage::KILLER_1;
                break;
            }

            case MovePickStage::KILLER_1: {
                m_stage = MovePickStage::KILLER_2;

                if (!m_skip_quiets && m_killer1 != chess::Move() &&
                    m_killer1 != m_ctx.tt_move) {
                    ensureLegal();
                    if (isValid(m_killer1)) {
                        bool is_capture = m_board.at(m_killer1.to()) != chess::Piece::NONE ||
                                          m_killer1.typeOf() == chess::Move::ENPASSANT ||
                                          m_killer1.typeOf() == chess::Move::PROMOTION;
                        if (!is_capture) {
                            m_last_score = 1500000;
                            is_quiet_out = true;
                            return m_killer1;
                        }
                    }
                }
                break;
            }

            case MovePickStage::KILLER_2: {
                m_stage = MovePickStage::COUNTER_MOVE;

                if (!m_skip_quiets && m_killer2 != chess::Move() &&
                    m_killer2 != m_ctx.tt_move && m_killer2 != m_killer1) {
                    ensureLegal();
                    if (isValid(m_killer2)) {
                        bool is_capture = m_board.at(m_killer2.to()) != chess::Piece::NONE ||
                                          m_killer2.typeOf() == chess::Move::ENPASSANT ||
                                          m_killer2.typeOf() == chess::Move::PROMOTION;
                        if (!is_capture) {
                            m_last_score = 1490000;
                            is_quiet_out = true;
                            return m_killer2;
                        }
                    }
                }
                break;
            }

            case MovePickStage::COUNTER_MOVE: {
                m_stage = MovePickStage::GENERATE_QUIETS;

                if (!m_skip_quiets && m_ctx.counter_move != chess::Move() &&
                    m_ctx.counter_move != m_ctx.tt_move &&
                    m_ctx.counter_move != m_killer1 &&
                    m_ctx.counter_move != m_killer2) {
                    ensureLegal();
                    if (isValid(m_ctx.counter_move)) {
                        bool is_capture = m_board.at(m_ctx.counter_move.to()) != chess::Piece::NONE ||
                                          m_ctx.counter_move.typeOf() == chess::Move::ENPASSANT ||
                                          m_ctx.counter_move.typeOf() == chess::Move::PROMOTION;
                        if (!is_capture) {
                            m_last_score = 1250000;
                            is_quiet_out = true;
                            return m_ctx.counter_move;
                        }
                    }
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
                while (m_quiet_idx < m_quiet_count) {
                    int best = m_quiet_idx;
                    for (int j = m_quiet_idx + 1; j < m_quiet_count; ++j) {
                        if (m_quiets[j].score > m_quiets[best].score) {
                            best = j;
                        }
                    }
                    if (best != m_quiet_idx) {
                        std::swap(m_quiets[m_quiet_idx], m_quiets[best]);
                    }

                    chess::Move move = m_quiets[m_quiet_idx].move;
                    int score = m_quiets[m_quiet_idx].score;
                    ++m_quiet_idx;

                    m_last_score = score;
                    is_quiet_out = true;
                    return move;
                }

                m_stage = MovePickStage::BAD_CAPTURES;
                break;
            }

            case MovePickStage::BAD_CAPTURES: {
                while (m_bad_capture_idx < m_bad_capture_count) {
                    chess::Move move = m_bad_captures[m_bad_capture_idx].move;
                    int score = m_bad_captures[m_bad_capture_idx].score;
                    ++m_bad_capture_idx;

                    m_last_score = -1000000 + score;
                    is_quiet_out = false;
                    return move;
                }

                m_stage = MovePickStage::DONE;
                break;
            }

            case MovePickStage::DONE:
                return chess::Move();
        }
    }
}

// ============================================================================
// QSearchMovePicker
// ============================================================================

QSearchMovePicker::QSearchMovePicker(const chess::Board& board, chess::Move tt_move, bool in_check)
    : m_board(board), m_tt_move(tt_move), m_in_check(in_check),
      m_stage(QMovePickStage::TT_MOVE),
      m_move_count(0), m_move_idx(0),
      m_last_score(0), m_returned_count(0),
      m_legal_generated(false) {
}

bool QSearchMovePicker::wasReturned(const chess::Move& move) const {
    for (int i = 0; i < m_returned_count; ++i) {
        if (m_returned[i] == move) return true;
    }
    return false;
}

void QSearchMovePicker::markReturned(const chess::Move& move) {
    if (m_returned_count < 256) {
        m_returned[m_returned_count++] = move;
    }
}

void QSearchMovePicker::ensureLegal() {
    if (!m_legal_generated) {
        chess::movegen::legalmoves(m_legal, m_board);
        m_legal_generated = true;
    }
}

bool QSearchMovePicker::isValid(const chess::Move& move) const {
    if (move == chess::Move()) return false;
    for (const auto& m : m_legal) {
        if (m == move) return true;
    }
    return false;
}

void QSearchMovePicker::pickBest(ScoredMove* moves, int start, int end) {
    int best = start;
    for (int j = start + 1; j < end; ++j) {
        if (moves[j].score > moves[best].score) {
            best = j;
        }
    }
    if (best != start) {
        std::swap(moves[start], moves[best]);
    }
}

void QSearchMovePicker::scoreCaptures() {
    ensureLegal();

    auto tacticalScore = [this](const chess::Move& move) -> int {
        chess::Piece attackerP = m_board.at(move.from());
        chess::Piece capturedP = chess::Piece::NONE;

        if (move.typeOf() == chess::Move::ENPASSANT) {
            chess::Square capSq(move.to().file(), move.from().rank());
            capturedP = m_board.at(capSq);
        } else {
            capturedP = m_board.at(move.to());
        }

        int victim = 0;
        if (capturedP != chess::Piece::NONE) {
            victim = pieceValue(capturedP.type());
        }

        int attacker = pieceValue(attackerP.type());
        int score = victim * 10 - attacker;

        if (capturedP != chess::Piece::NONE) {
            score += g_captureHistory.get(
                static_cast<int>(attackerP.type()),
                move.to().index(),
                static_cast<int>(capturedP.type())) / 16;
        }

        if (move.typeOf() == chess::Move::PROMOTION) {
            score += 600000 + pieceValue(move.promotionType());
        }

        return score;
    };

    m_move_count = 0;
    m_move_idx = 0;

    for (const auto& move : m_legal) {
        if (move == m_tt_move) continue;
        if (m_move_count >= 256) break;

        if (m_in_check) {
            bool is_capture = m_board.at(move.to()) != chess::Piece::NONE ||
                              move.typeOf() == chess::Move::ENPASSANT;
            bool is_promo = move.typeOf() == chess::Move::PROMOTION;

            int score = 0;
            if (is_capture || is_promo) {
                score = 2000000 + tacticalScore(move);
            }

            m_moves[m_move_count].move = move;
            m_moves[m_move_count].score = score;
            ++m_move_count;
        } else {
            bool is_tactical = m_board.at(move.to()) != chess::Piece::NONE ||
                               move.typeOf() == chess::Move::PROMOTION ||
                               move.typeOf() == chess::Move::ENPASSANT;
            if (!is_tactical) continue;

            m_moves[m_move_count].move = move;
            m_moves[m_move_count].score = tacticalScore(move);
            ++m_move_count;
        }
    }
}

chess::Move QSearchMovePicker::next() {
    while (true) {
        switch (m_stage) {
            case QMovePickStage::TT_MOVE: {
                m_stage = QMovePickStage::GENERATE_CAPTURES;

                if (m_tt_move != chess::Move()) {
                    ensureLegal();
                    if (isValid(m_tt_move)) {
                        if (!m_in_check) {
                            bool is_tactical = m_board.at(m_tt_move.to()) != chess::Piece::NONE ||
                                               m_tt_move.typeOf() == chess::Move::PROMOTION ||
                                               m_tt_move.typeOf() == chess::Move::ENPASSANT;
                            if (!is_tactical) break;
                        }

                        m_last_score = 3000000;
                        return m_tt_move;
                    }
                }
                break;
            }

            case QMovePickStage::GENERATE_CAPTURES: {
                scoreCaptures();
                m_stage = QMovePickStage::CAPTURES;
                break;
            }

            case QMovePickStage::CAPTURES: {
                while (m_move_idx < m_move_count) {
                    pickBest(m_moves, m_move_idx, m_move_count);

                    chess::Move move = m_moves[m_move_idx].move;
                    m_last_score = m_moves[m_move_idx].score;
                    ++m_move_idx;

                    return move;
                }

                m_stage = QMovePickStage::DONE;
                break;
            }

            case QMovePickStage::DONE:
                return chess::Move();
        }
    }
}

// ============================================================================
// Legacy helpers
// ============================================================================

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move,
                         const MovePickerContext& ctx) {
    if (move == ctx.tt_move && ctx.tt_move != chess::Move()) {
        return 1000000;
    }

    if (move.typeOf() == chess::Move::PROMOTION) {
        return 900000 + pieceValue(move.promotionType());
    }

    chess::Piece attacker = board.at(move.from());
    chess::Piece captured = chess::Piece::NONE;

    if (move.typeOf() == chess::Move::ENPASSANT) {
        chess::Square capSq(move.to().file(), move.from().rank());
        captured = board.at(capSq);
    } else {
        captured = board.at(move.to());
    }

    bool is_tactical = captured != chess::Piece::NONE ||
                       move.typeOf() == chess::Move::ENPASSANT ||
                       move.typeOf() == chess::Move::PROMOTION;

    if (is_tactical) {
        int see_score = chess::see::see_ge(board, move, 0) ? 100
                      : (chess::see::see_ge(board, move, -50) ? 0 : -100);

        int victimValue = captured != chess::Piece::NONE ? pieceValue(captured.type()) : 100;
        int attackerValue = pieceValue(attacker.type());

        int cap_hist = 0;
        if (captured != chess::Piece::NONE) {
            cap_hist = g_captureHistory.get(
                static_cast<int>(attacker.type()),
                move.to().index(),
                static_cast<int>(captured.type())) / 32;
        }

        return 800000 + see_score * 1000 + victimValue * 10 - attackerValue + cap_hist;
    }

    int killer_score = g_killerMoves.get_killer_score(ctx.ply, move);
    if (killer_score > 0) {
        return 700000 + killer_score * 1000;
    }

    if (move == ctx.counter_move && ctx.counter_move != chess::Move()) {
        return 650000;
    }

    int score = g_butterflyHistory.get(ctx.side_to_move, move.from(), move.to());
    if (move.typeOf() == chess::Move::CASTLING) {
        score += 50;
    }

    return score;
}

std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves,
                                   const chess::Board& board,
                                   const MovePickerContext& ctx) {
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());

    for (const auto& move : moves) {
        scored.emplace_back(move, scoreMoveForOrdering(board, move, ctx));
    }

    return scored;
}

void pickNextMove(std::vector<ScoredMove>& moves, size_t current) {
    if (current >= moves.size()) return;

    size_t best_idx = current;
    int best_score = moves[current].score;

    for (size_t i = current + 1; i < moves.size(); ++i) {
        if (moves[i].score > best_score) {
            best_score = moves[i].score;
            best_idx = i;
        }
    }

    if (best_idx != current) {
        std::swap(moves[current], moves[best_idx]);
    }
}