#pragma once
#include "chess.hpp"
#include "types.h"
#include <vector>

// Forward declare — full definition lives in policy.h
struct NodePolicy;

struct MovePickerContext {
    chess::Move tt_move{};
    chess::Move counter_move{};
    chess::Color side_to_move{};
    int ply = 0;
    SearchStack* ss = nullptr;
    const NodePolicy* node_policy = nullptr;   // in-tree policy (nullable)

    MovePickerContext() = default;
    MovePickerContext(chess::Move tt, chess::Move counter, chess::Color side,
                      int ply_, SearchStack* ss_,
                      const NodePolicy* np = nullptr)
        : tt_move(tt), counter_move(counter), side_to_move(side),
          ply(ply_), ss(ss_), node_policy(np) {}
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

    // Skip remaining quiets without killing bad captures (for LMP).
    void skipQuiets() {
        if (m_stage == MovePickStage::GENERATE_QUIETS ||
            m_stage == MovePickStage::QUIETS) {
            m_stage = MovePickStage::BAD_CAPTURES;
        }
    }

    // Public accessors so search can generate moves once and share the
    // list with computeNodePolicy (avoids double legalmoves() call).
    void ensureLegalPublic() { ensureLegal(); }
    const chess::Movelist& legalMoves() const { return m_all_legal; }

    // Inject a pre-computed node policy after construction.
    // Must be called before the first next() that reaches QUIETS stage.
    void setNodePolicy(const NodePolicy* np) { m_ctx.node_policy = np; }

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

    chess::Movelist m_all_legal;
    bool m_legal_generated = false;

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

    chess::Movelist m_legal;
    bool m_legal_generated = false;

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