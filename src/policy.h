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
// Root LMR: lift-based continuous adjustment
// ============================================================================
// For each quiet root move, compute the "lift" signal:
//
//   p_quiet(m) = softmax over quiets only
//   rel(m)     = ln( p_quiet(m) * nq )
//
// Properties:
//   - Uniform position (all quiets equal): rel = 0 for every move → adj = 0
//     (self-calibrating: flat policy → zero distortion, no entropy gate needed)
//   - Sharp position: favorite gets rel >> 0, tail gets rel << 0
//   - By Jensen's inequality E[rel] <= 0, so the default bias is mildly
//     "reduce more", with a few moves pulled negative ("reduce less")
//
// LMR adjustment:
//   adj = clamp( round( -POLICY_LMR_K * rel ), MIN_ADJ, MAX_ADJ )
//
// With K = 0.75:
//   rel = +2.1  (dominant favorite, ~80% of 10 quiets) → adj = -2
//   rel = +1.1  (strong favorite,  ~30%)               → adj = -1
//   rel =  0.0  (average quiet)                        → adj =  0
//   rel = -1.2  (below average,     ~3%)               → adj = +1
//   rel = -3.0  (terrible,         ~0.5%)              → adj = +2
//   rel = -4.6  (garbage,          ~0.1%)              → adj = +3
//
// SPSA-tune POLICY_LMR_K. Start at 0.75.
constexpr int   POLICY_ROOT_LMR_MIN_DEPTH = 3;
constexpr float POLICY_LMR_K              = 0.75f;
constexpr int   POLICY_LMR_MIN_ADJ        = -2;
constexpr int   POLICY_LMR_MAX_ADJ        = 3;
constexpr int   POLICY_LMR_MIN_QUIETS     = 2;
constexpr float POLICY_REL_NONE           = -1000.0f;

// Legacy
constexpr int POLICY_QUIET_WEIGHT  = 1024;
constexpr int POLICY_MIN_DEPTH     = 6;

// ============================================================================
// 1a: policy-disagreement time management (root only, zero NPS tax in-tree)
// ============================================================================
constexpr int    POLICY_TM_MIN_DEPTH    = 6;
constexpr float  POLICY_TM_AGREE_CONF   = 0.35f;
constexpr float  POLICY_TM_UNCERTAIN    = 0.18f;
constexpr double POLICY_TM_DISAGREE     = 1.35;
constexpr double POLICY_TM_UNCERTAIN_S  = 1.25;
constexpr double POLICY_TM_AGREE_S      = 0.88;

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

    bool rankLegalQuiets(const chess::Board& board,
                         const chess::Movelist& moves,
                         int* out_rank,
                         int* out_nq = nullptr) const;

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