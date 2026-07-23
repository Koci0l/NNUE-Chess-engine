#pragma once

#include "chess.hpp"
#include "types.h"

#include <vector>

struct MovePickerContext {
    chess::Move tt_move{};
    chess::Move counter_move{};
    chess::Color side_to_move{};
    int ply = 0;
    SearchStack* ss = nullptr;

    bool root_policy_order = false;
    const chess::Movelist* policy_moves = nullptr;
    const int* policy_bonus = nullptr;

    MovePickerContext() = default;

    MovePickerContext(chess::Move tt, chess::Move counter, chess::Color side,
                      int ply_, SearchStack* ss_)
        : tt_move(tt),
          counter_move(counter),
          side_to_move(side),
          ply(ply_),
          ss(ss_) {}
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

class MovePicker {
public:
    MovePicker(const chess::Board& board, const MovePickerContext& ctx,
               int depth, bool skip_quiets, bool use_policy_unused = false);

    chess::Move next(bool& is_quiet_out);
    int lastScore() const { return m_last_score; }

private:
    const chess::Board& m_board;
    MovePickerContext m_ctx;
    int m_depth;
    bool m_skip_quiets;

    MovePickStage m_stage;

    chess::Move m_killer1{};
    chess::Move m_killer2{};

    ScoredMove m_captures[256];
    int m_capture_count = 0;
    int m_capture_idx = 0;

    ScoredMove m_bad_captures[256];
    int m_bad_capture_count = 0;
    int m_bad_capture_idx = 0;

    ScoredMove m_quiets[256];
    int m_quiet_count = 0;
    int m_quiet_idx = 0;

    int m_last_score = 0;
    chess::Move m_returned[512];
    int m_returned_count = 0;

    chess::Movelist m_all_legal;
    bool m_legal_generated = false;

    bool wasReturned(const chess::Move& move) const;
    void markReturned(const chess::Move& move);
    void ensureLegal();
    bool isValid(const chess::Move& move) const;
    int scoreOneCapture(const chess::Move& move);
    int scoreOneQuiet(const chess::Move& move);
    void scoreCaptures();
    void scoreQuiets();
};

enum class QMovePickStage {
    TT_MOVE,
    GENERATE_CAPTURES,
    CAPTURES,
    DONE
};

class QSearchMovePicker {
public:
    QSearchMovePicker(const chess::Board& board, chess::Move tt_move, bool in_check);
    chess::Move next();
    int lastScore() const { return m_last_score; }

private:
    const chess::Board& m_board;
    chess::Move m_tt_move;
    bool m_in_check;
    QMovePickStage m_stage;

    ScoredMove m_moves[256];
    int m_move_count = 0;
    int m_move_idx = 0;
    int m_last_score = 0;

    chess::Move m_returned[256];
    int m_returned_count = 0;

    chess::Movelist m_legal;
    bool m_legal_generated = false;

    bool wasReturned(const chess::Move& move) const;
    void markReturned(const chess::Move& move);
    void ensureLegal();
    bool isValid(const chess::Move& move) const;
    void pickBest(ScoredMove* moves, int start, int end);
    void scoreCaptures();
};

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move,
                         const MovePickerContext& ctx);

std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves,
                                   const chess::Board& board,
                                   const MovePickerContext& ctx);

void pickNextMove(std::vector<ScoredMove>& moves, size_t current);