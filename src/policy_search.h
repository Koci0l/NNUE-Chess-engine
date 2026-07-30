#pragma once
// ============================================================================
// policy_search.h
//
// Selective, cached policy evaluation for use *inside* the search tree.
//
// The 2048-wide policy net is far too expensive to call at every node
// (~2.1M FMA in layer 1 alone, ~12 MB of weights streamed per call). This
// module makes it affordable by:
//
//   1. Gating: only real PV nodes, depth >= g_policySearchMinDepth, and
//      ply <= g_policySearchMaxPly are ever considered.
//   2. Memoization: every evaluation is stored in a per-thread, direct
//      mapped, zobrist-keyed cache. Policy output is a pure function of the
//      position, so entries never expire and are reused across aspiration
//      re-searches, iterative-deepening iterations, and transpositions.
//   3. Precomputation: the ordering bonus and the LMR reduction delta are
//      baked into the cache entry at fill time, so consumers only ever do a
//      cheap 16-bit linear scan.
//
// Everything is runtime-tunable via the globals below (wire to UCI/SPSA).
// ============================================================================

#include "chess.hpp"
#include "policy.h"

#include <cstdint>

// ----------------------------------------------------------------------------
// Tunables (defaults; the g_* globals are what the search actually reads)
// ----------------------------------------------------------------------------

constexpr int   POLICY_SEARCH_MIN_DEPTH_DEFAULT = 4;     // min depth to probe
constexpr int   POLICY_SEARCH_MAX_PLY_DEFAULT   = 8;     // max ply from root
constexpr int   POLICY_ORDER_SCALE_DEFAULT      = 1200;  // bonus per unit rel
constexpr int   POLICY_SEED_MIN_DEPTH_DEFAULT   = 5;     // TT-seed min depth
constexpr float POLICY_SEED_MIN_PROB_DEFAULT    = 0.25f; // TT-seed confidence

// Number of quiets tracked per cache entry. Quiets beyond this get no bonus
// and no LMR adjustment, which is the correct conservative fallback.
constexpr int POLICY_QCACHE_MOVES = 64;

// Direct-mapped cache size, per thread. 2048 * ~416B ~= 850 KB / thread.
constexpr int POLICY_QCACHE_BITS = 11;
constexpr int POLICY_QCACHE_SIZE = 1 << POLICY_QCACHE_BITS;
constexpr int POLICY_QCACHE_MASK = POLICY_QCACHE_SIZE - 1;

// Bonus clamp. Kept deliberately smaller than the typical magnitude of
// (butterfly + cont1 + cont2) so policy acts as a strong tiebreaker rather
// than overriding well-established history.
constexpr int POLICY_ORDER_BONUS_MIN = -6000;
constexpr int POLICY_ORDER_BONUS_MAX =  8000;

// Damping: policy bonus is scaled by K / (K + |history|). Set to 0 to disable.
constexpr int POLICY_ORDER_DAMP_K = 6000;

// ----------------------------------------------------------------------------
// Runtime knobs
// ----------------------------------------------------------------------------

extern bool  g_policyInSearch;          // master on/off switch
extern int   g_policySearchMinDepth;
extern int   g_policySearchMaxPly;
extern int   g_policyOrderScale;
extern int   g_policySeedMinDepth;
extern float g_policySeedMinProb;
extern bool  g_policyUseOrdering;       // enable quiet ordering bonus
extern bool  g_policyUseLMR;            // enable LMR delta
extern bool  g_policyUseSeed;           // enable TT-move seeding

// ----------------------------------------------------------------------------
// Cache entry
// ----------------------------------------------------------------------------

struct PolicyQuiets {
    uint64_t key = 0;          // zobrist; 0 == empty slot

    int32_t  n = 0;            // number of quiets stored (<= POLICY_QCACHE_MOVES)
    int32_t  nq_total = 0;     // total legal quiets in the position

    float    sharpness = 0.0f; // 0..1 confidence in the quiet distribution

    uint16_t top_any = 0;      // best move over *all* legals (raw 16-bit)
    float    top_any_prob = 0.0f;

    // Parallel arrays, sorted by descending policy probability.
    uint16_t mv[POLICY_QCACHE_MOVES];
    int16_t  bonus[POLICY_QCACHE_MOVES];      // move-ordering bonus
    int8_t   lmr_delta[POLICY_QCACHE_MOVES];  // add to LMR reduction
    int8_t   rank[POLICY_QCACHE_MOVES];       // 0 == most plausible quiet

    inline int find(const chess::Move& m) const {
        const uint16_t raw = m.move();
        for (int i = 0; i < n; ++i) {
            if (mv[i] == raw) return i;
        }
        return -1;
    }

    inline int bonusFor(const chess::Move& m) const {
        const int i = find(m);
        return (i < 0) ? 0 : int(bonus[i]);
    }

    inline int lmrDeltaFor(const chess::Move& m) const {
        const int i = find(m);
        return (i < 0) ? 0 : int(lmr_delta[i]);
    }

    inline int rankOf(const chess::Move& m) const {
        const int i = find(m);
        return (i < 0) ? -1 : int(rank[i]);
    }

    inline bool isTopQuiet(const chess::Move& m) const {
        return n > 0 && mv[0] == m.move();
    }
};

// ----------------------------------------------------------------------------
// API
// ----------------------------------------------------------------------------

// Cheap, branch-only gate. Call this before doing anything else; it must be
// true before policyProbeQuiets() is worth calling.
inline bool policyGateOK(bool is_pv_node, bool in_check, bool in_singular,
                         int depth, int ply_from_root) {
    return g_policyInSearch
        && g_policy.loaded
        && is_pv_node
        && !in_check
        && !in_singular
        && depth >= g_policySearchMinDepth
        && ply_from_root <= g_policySearchMaxPly;
}

// Returns a cached (or freshly computed) policy summary for `board`.
// Returns nullptr if the net is not loaded, the position has no quiets, or
// the evaluation failed. Safe to call repeatedly; only the first call for a
// given zobrist actually runs the network.
const PolicyQuiets* policyProbeQuiets(const chess::Board& board, uint64_t hash);

// Per-thread instrumentation.
void policyCacheStats(uint64_t& probes, uint64_t& hits, uint64_t& evals);
void policyCacheResetStats();
void policyCacheClear();