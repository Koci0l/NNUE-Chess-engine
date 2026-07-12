#pragma once

#include "chess.hpp"
#include "types.h"
#include "policy.h"

#include <vector>

struct MovePickerContext {
    chess::Move tt_move;
    chess::Move counter_move;
    chess::Color side_to_move;
    int ply;
    const SearchStack* ss;

    MovePickerContext(chess::Move tt, chess::Move counter, chess::Color stm, int p,
                      const SearchStack* stack = nullptr)
        : tt_move(tt), counter_move(counter), side_to_move(stm), ply(p), ss(stack) {}
};

enum class MovePickStage {
    TT_MOVE,
    GENERATE_CAPTURES,
    GOOD_CAPTURES,
    KILLER_1,
    KILLER_2,
    COUNTER_MOVE,
    GENERATE_QUIETS,
    QUIETS,
    BAD_CAPTURES,
    DONE
};

enum class QMovePickStage {
    TT_MOVE,
    GENERATE_CAPTURES,
    CAPTURES,
    DONE
};

class MovePicker {
public:
    MovePicker(const chess::Board& board, const MovePickerContext& ctx,
               int depth, bool skip_quiets = false);

    chess::Move next(bool& is_quiet_out);
    int last_score() const { return m_last_score; }

private:
    void scoreCaptures();
    void scoreQuiets();
    int scoreOneCapture(const chess::Move& move);
    int scoreOneQuiet(const chess::Move& move);
    bool isValid(const chess::Move& move) const;
    void ensureLegal();
    void ensurePolicy();

    bool wasReturned(const chess::Move& move) const;
    void markReturned(const chess::Move& move);

    const chess::Board& m_board;
    MovePickerContext m_ctx;
    int m_depth;
    bool m_skip_quiets;

    MovePickStage m_stage;

    ScoredMove m_captures[256];
    int m_capture_count;
    int m_capture_idx;

    ScoredMove m_bad_captures[256];
    int m_bad_capture_count;
    int m_bad_capture_idx;

    ScoredMove m_quiets[256];
    int m_quiet_count;
    int m_quiet_idx;

    chess::Move m_killer1;
    chess::Move m_killer2;

    int m_last_score;

    chess::Move m_returned[512];
    int m_returned_count;

    chess::Movelist m_all_legal;
    bool m_legal_generated;

    // Policy cache for this node (aligned with m_all_legal indices)
    float m_policy_bonus[256];
    bool  m_policy_ready;
};

class QSearchMovePicker {
public:
    QSearchMovePicker(const chess::Board& board, chess::Move tt_move, bool in_check);

    chess::Move next();
    int last_score() const { return m_last_score; }

private:
    void scoreCaptures();
    void pickBest(ScoredMove* moves, int start, int end);
    bool isValid(const chess::Move& move) const;
    void ensureLegal();

    bool wasReturned(const chess::Move& move) const;
    void markReturned(const chess::Move& move);

    const chess::Board& m_board;
    chess::Move m_tt_move;
    bool m_in_check;

    QMovePickStage m_stage;

    ScoredMove m_moves[256];
    int m_move_count;
    int m_move_idx;

    int m_last_score;

    chess::Move m_returned[256];
    int m_returned_count;

    chess::Movelist m_legal;
    bool m_legal_generated;
};

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move,
                         const MovePickerContext& ctx);

std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves,
                                   const chess::Board& board,
                                   const MovePickerContext& ctx);

void pickNextMove(std::vector<ScoredMove>& moves, size_t current);