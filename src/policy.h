#pragma once

#include "chess.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// Matches monty/bullet policy trainer (inputs.rs + model.rs)
// ============================================================================

constexpr int POLICY_PLANE       = 768;
constexpr int POLICY_INPUT_SIZE  = POLICY_PLANE * 4; // 3072
constexpr int POLICY_HL          = 128;
constexpr int POLICY_HL_PAIR     = POLICY_HL / 2;    // 64
constexpr int POLICY_MAX_ACTIVE  = 32;
constexpr int POLICY_QA          = 128;
constexpr int POLICY_PROMOS      = 4 * 22;           // 88
constexpr int POLICY_SEE_TH      = -108;

// Quiet ordering blend: bonus in [0, POLICY_QUIET_WEIGHT]
constexpr int POLICY_QUIET_WEIGHT = 2048;

// Only run policy in move picker at this depth and above (huge NPS win)
constexpr int POLICY_MIN_DEPTH = 6;

struct PolicyNet {
    bool loaded = false;

    int from_to   = 0;
    int num_moves = 0;

    std::vector<float> l0w;
    std::vector<float> l0b;
    std::vector<float> l1w;
    std::vector<float> l1b;

    uint64_t destinations[64][6]{};
    int      offsets[6][65]{};

    // quantised.bin L1 is [out][in] (verified vs trainer)
    bool l1_out_major = true;

    PolicyNet();

    void clear();
    bool loadFromMemory(const std::uint8_t* data, std::size_t size,
                        const char* label = "memory");
    bool load(const std::string& path);

    // Full path (debug / UCI): logits + softmax over all legal moves
    bool scoreLegalMoves(const chess::Board& board,
                         const chess::Movelist& moves,
                         float* out_probs) const;

    bool logitsLegalMoves(const chess::Board& board,
                          const chess::Movelist& moves,
                          float* out_logits) const;

    // Fast AB path: quiets only, no softmax.
    // out_bonus[i] aligned with moves[]; 0 for captures/promos;
    // for quiets in [0, POLICY_QUIET_WEIGHT] by logit rank span.
    bool scoreQuietsForOrdering(const chess::Board& board,
                                const chess::Movelist& moves,
                                float* out_bonus) const;

    void collectFeatures(const chess::Board& board, int* feats, int& nfeats) const;
    int  mapMoveToIndex(const chess::Board& board, const chess::Move& m) const;

    void debugPosition(const chess::Board& board, int topN = 16) const;
    void debugMove(const chess::Board& board, const chess::Move& m) const;

    static int      stmKingIndex(const chess::Board& board);
    static int      flipMask(const chess::Board& board);
    static uint64_t attacksBySide(const chess::Board& board, chess::Color side);
};

extern PolicyNet g_policy;