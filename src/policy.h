// ============================================================================
// policy.h
// ============================================================================
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

// ============================================================================
// Root LMR: continuous logit-gap adjustment
// ============================================================================
// For each quiet root move:
//   logit_gap = max_quiet_logit - move_logit          (>= 0)
//   adj       = round(POLICY_LMR_SCALE * logit_gap - POLICY_LMR_OFFSET)
//   adj       = clamp(adj, POLICY_LMR_MIN_ADJ, POLICY_LMR_MAX_ADJ)
//   reduction += adj
//
// With the defaults below:
//   gap = 0   -> -1  (policy top quiet is protected)
//   gap = 2   ->  0  (neutral)
//   gap = 4-6 -> +1  (reduced more)
//   gap >= 8  -> +2  (reduced most, clamped)
//
// Tune POLICY_LMR_SCALE and POLICY_LMR_OFFSET via SPRT.
constexpr int   POLICY_ROOT_LMR_MIN_DEPTH = 3;
constexpr float POLICY_LMR_SCALE          = 0.25f;
constexpr float POLICY_LMR_OFFSET         = 0.50f;
constexpr int   POLICY_LMR_MIN_ADJ        = -1;
constexpr int   POLICY_LMR_MAX_ADJ        = 2;
constexpr int   POLICY_LMR_MIN_QUIETS     = 4;

// Legacy
constexpr int POLICY_QUIET_WEIGHT  = 1024;
constexpr int POLICY_MIN_DEPTH     = 6;

constexpr int    POLICY_TM_MIN_DEPTH    = 6;
constexpr float  POLICY_TM_AGREE_CONF   = 0.35f;  // top1 >= this + agree -> less time
constexpr float  POLICY_TM_UNCERTAIN    = 0.18f;  // top1 <  this         -> more time
constexpr double POLICY_TM_DISAGREE     = 1.35;   // policy top1 != search best
constexpr double POLICY_TM_UNCERTAIN_S  = 1.25;   // low confidence
constexpr double POLICY_TM_AGREE_S      = 0.88;   // high-conf agreement

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
    bool l1_out_major = true;

    PolicyNet();
    void clear();

    bool loadFromMemory(const std::uint8_t* data, std::size_t size,
                        const char* label = "memory");
    bool load(const std::string& path);

    bool scoreLegalMoves(const chess::Board& board,
                         const chess::Movelist& moves,
                         float* out_probs) const;

    bool logitsLegalMoves(const chess::Board& board,
                          const chess::Movelist& moves,
                          float* out_logits) const;

    // out_rank[i] aligned with moves[]:
    //   -1 = capture / promo / failed index
    //    0 .. nq-1 = quiet rank (0 best)
    // *out_nq = number of quiets ranked (optional)
    bool rankLegalQuiets(const chess::Board& board,
                         const chess::Movelist& moves,
                         int* out_rank,
                         int* out_nq = nullptr) const;

    // On success writes out_top / out_top1_prob; entropy_out optional.
    bool rootAdvice(const chess::Board& board,
                    chess::Move& out_top,
                    float& out_top1_prob,
                    float* entropy_out = nullptr) const;

    void collectFeatures(const chess::Board& board, int* feats, int& nfeats) const;
    int  mapMoveToIndex(const chess::Board& board, const chess::Move& m) const;

    void debugPosition(const chess::Board& board, int topN = 16) const;
    void debugMove(const chess::Board& board, const chess::Move& m) const;

    static int      stmKingIndex(const chess::Board& board);
    static int      flipMask(const chess::Board& board);
    static uint64_t attacksBySide(const chess::Board& board, chess::Color side);
};

extern PolicyNet g_policy;