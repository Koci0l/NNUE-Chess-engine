#pragma once

#include "chess.hpp"

#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// Matches monty/bullet policy trainer (inputs.rs + model.rs)
// ============================================================================
// INPUT_SIZE        = 768 * 4 = 3072
// HL                = 128
// after pairwise    = 64
// NUM_MOVES_INDICES = 2 * FROM_TO
// quant             = i8 / 128
// ============================================================================

constexpr int POLICY_PLANE       = 768;
constexpr int POLICY_INPUT_SIZE  = POLICY_PLANE * 4; // 3072
constexpr int POLICY_HL          = 128;
constexpr int POLICY_HL_PAIR     = POLICY_HL / 2;    // 64
constexpr int POLICY_MAX_ACTIVE  = 32;
constexpr int POLICY_QA          = 128;
constexpr int POLICY_PROMOS      = 4 * 22;           // 88
constexpr int POLICY_SEE_TH      = -108;

// Quiet move-ordering blend only (safe first integration).
// score += int(prob * POLICY_QUIET_WEIGHT)
constexpr int POLICY_QUIET_WEIGHT = 2048;

struct PolicyNet {
    bool loaded = false;

    // Computed from DESTINATIONS / OFFSETS (same as Rust)
    int from_to   = 0;
    int num_moves = 0; // 2 * from_to

    std::vector<float> l0w; // [INPUT_SIZE * HL]
    std::vector<float> l0b; // [HL]
    std::vector<float> l1w; // [HL_PAIR * NUM_MOVES] storage size
    std::vector<float> l1b; // [NUM_MOVES]

    // [sq][pc]  pc: P=0,N=1,B=2,R=3,Q=4,K=5
    uint64_t destinations[64][6]{};
    int      offsets[6][65]{};

    // quantised.bin L1 is [out][in]  -> mi * HL_PAIR + k
    // Verified against trainer eval on startpos.
    bool l1_out_major = true;

    PolicyNet();

    void clear();
    bool load(const std::string& path);

    bool scoreLegalMoves(const chess::Board& board,
                         const chess::Movelist& moves,
                         float* out_probs) const;

    bool logitsLegalMoves(const chess::Board& board,
                          const chess::Movelist& moves,
                          float* out_logits) const;

    void collectFeatures(const chess::Board& board, int* feats, int& nfeats) const;
    int  mapMoveToIndex(const chess::Board& board, const chess::Move& m) const;

    void debugPosition(const chess::Board& board, int topN = 16) const;
    void debugMove(const chess::Board& board, const chess::Move& m) const;

    static int      stmKingIndex(const chess::Board& board);
    static int      flipMask(const chess::Board& board);
    static uint64_t attacksBySide(const chess::Board& board, chess::Color side);
};

extern PolicyNet g_policy;