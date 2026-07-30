// ============================================================================
// policy_search.cpp
// ============================================================================
#include "policy_search.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ----------------------------------------------------------------------------
// Runtime knobs
// ----------------------------------------------------------------------------

bool  g_policyInSearch       = true;
int   g_policySearchMinDepth = POLICY_SEARCH_MIN_DEPTH_DEFAULT;
int   g_policySearchMaxPly   = POLICY_SEARCH_MAX_PLY_DEFAULT;
int   g_policyOrderScale     = POLICY_ORDER_SCALE_DEFAULT;
int   g_policySeedMinDepth   = POLICY_SEED_MIN_DEPTH_DEFAULT;
float g_policySeedMinProb    = POLICY_SEED_MIN_PROB_DEFAULT;

bool  g_policyUseOrdering    = false;
bool  g_policyUseLMR         = false;
bool  g_policyUseSeed        = true;

// ----------------------------------------------------------------------------
// Per-thread cache
// ----------------------------------------------------------------------------

static thread_local std::vector<PolicyQuiets> tl_cache;

static thread_local uint64_t tl_probes = 0;
static thread_local uint64_t tl_hits   = 0;
static thread_local uint64_t tl_evals  = 0;

static inline PolicyQuiets* cacheSlot(uint64_t hash) {
    if (tl_cache.empty()) {
        tl_cache.assign(POLICY_QCACHE_SIZE, PolicyQuiets{});
    }
    return &tl_cache[static_cast<size_t>(hash & POLICY_QCACHE_MASK)];
}

void policyCacheStats(uint64_t& probes, uint64_t& hits, uint64_t& evals) {
    probes = tl_probes;
    hits   = tl_hits;
    evals  = tl_evals;
}

void policyCacheResetStats() {
    tl_probes = tl_hits = tl_evals = 0;
}

void policyCacheClear() {
    if (!tl_cache.empty()) {
        std::fill(tl_cache.begin(), tl_cache.end(), PolicyQuiets{});
    }
    policyCacheResetStats();
}

// ----------------------------------------------------------------------------
// Bonus / reduction shaping
//
// `rel` is log(p_quiet * nq): 0 means "exactly average plausibility",
// positive means "more plausible than a uniform pick", negative means less.
// This is the same normalisation used by computeRootPolicy(), so the search
// and the root agree on what "good" means.
// ----------------------------------------------------------------------------

static inline int shapeOrderBonus(float rel, int rank, float sharp) {
    const float rel_c = std::clamp(rel, -3.0f, 3.0f);

    int b = static_cast<int>(float(g_policyOrderScale) * rel_c * sharp);

    int rank_bonus = 0;
    if      (rank == 0) rank_bonus = 4000;
    else if (rank == 1) rank_bonus = 2200;
    else if (rank <= 3) rank_bonus = 1200;
    else if (rank <= 7) rank_bonus = 400;

    b += static_cast<int>(float(rank_bonus) * sharp);

    return std::clamp(b, POLICY_ORDER_BONUS_MIN, POLICY_ORDER_BONUS_MAX);
}

static inline int shapeLMRDelta(float rel, int rank, int nq, float sharp) {
    // Negative == reduce less (search deeper).
    float adj = -0.90f * std::clamp(rel, -2.5f, 2.5f);
    adj = std::clamp(adj, -2.0f, 2.0f);
    adj *= sharp;

    int d = static_cast<int>(std::lround(adj));

    // Hard floor for the clearly-best quiet in a sharp position.
    if (rank == 0 && sharp >= 0.50f && d > -1) d = -1;

    // Deep tail of a wide, sharp move list: reduce a bit harder.
    if (sharp >= 0.40f && nq >= 16 && rank >= 12 && d < 1) d += 1;

    return std::clamp(d, -2, 2);
}

// ----------------------------------------------------------------------------
// Probe
// ----------------------------------------------------------------------------

const PolicyQuiets* policyProbeQuiets(const chess::Board& board, uint64_t hash) {
    if (!g_policy.loaded) return nullptr;

    // Zobrist 0 is astronomically unlikely but would collide with "empty".
    if (hash == 0) hash = 1;

    ++tl_probes;

    PolicyQuiets* e = cacheSlot(hash);

    if (e->key == hash) {
        ++tl_hits;
        return (e->n > 0) ? e : nullptr;
    }

    // ---- Miss: run the network -------------------------------------------
    ++tl_evals;

    chess::Movelist legals;
    chess::movegen::legalmoves(legals, board);

    const int nlegal = static_cast<int>(legals.size());
    if (nlegal <= 0 || nlegal > 256) {
        return nullptr;
    }

    float logits[256];
    if (!g_policy.logitsLegalMoves(board, legals, logits)) {
        return nullptr;
    }

    // Claim the slot now; even a "no quiets" result is worth memoizing so we
    // never re-run the net on this position.
    e->key          = hash;
    e->n            = 0;
    e->nq_total     = 0;
    e->sharpness    = 0.0f;
    e->top_any      = 0;
    e->top_any_prob = 0.0f;

    // ---- Softmax over all legal moves (for TT-move seeding) --------------
    {
        float mx = -1e30f;
        for (int i = 0; i < nlegal; ++i) mx = std::max(mx, logits[i]);

        float sum = 0.0f;
        float pa[256];
        for (int i = 0; i < nlegal; ++i) {
            pa[i] = std::exp(logits[i] - mx);
            sum += pa[i];
        }
        if (sum <= 0.0f) sum = 1.0f;

        int   best_i = 0;
        float best_p = -1.0f;
        for (int i = 0; i < nlegal; ++i) {
            const float p = pa[i] / sum;
            if (p > best_p) { best_p = p; best_i = i; }
        }

        e->top_any      = legals[best_i].move();
        e->top_any_prob = best_p;
    }

    // ---- Quiet-only softmax ----------------------------------------------
    int   qidx[256];
    int   nq = 0;
    float max_q = -1e30f;

    for (int i = 0; i < nlegal; ++i) {
        if (!policyQuietLocal(board, legals[i])) continue;
        qidx[nq++] = i;
        max_q = std::max(max_q, logits[i]);
    }

    e->nq_total = nq;

    if (nq <= 0) {
        return nullptr;   // memoized as "no quiets"
    }

    float qp[256];
    float sum_q = 0.0f;
    for (int j = 0; j < nq; ++j) {
        qp[j] = std::exp(logits[qidx[j]] - max_q);
        sum_q += qp[j];
    }
    if (sum_q <= 0.0f) sum_q = 1.0f;
    for (int j = 0; j < nq; ++j) qp[j] /= sum_q;

    // Normalised entropy -> sharpness (identical shaping to computeRootPolicy)
    double qent = 0.0;
    for (int j = 0; j < nq; ++j) {
        if (qp[j] > 1e-12f) qent -= double(qp[j]) * std::log(double(qp[j]));
    }

    float norm_qent = 0.0f;
    if (nq > 1) {
        norm_qent = static_cast<float>(qent / std::log(static_cast<double>(nq)));
    }

    int order[256];
    for (int j = 0; j < nq; ++j) order[j] = j;
    std::sort(order, order + nq, [&](int a, int b) { return qp[a] > qp[b]; });

    float sharp = std::clamp((0.90f - norm_qent) / 0.35f, 0.0f, 1.0f);
    if (qp[order[0]] < 0.12f) sharp *= 0.5f;

    e->sharpness = sharp;

    const int store = std::min(nq, POLICY_QCACHE_MOVES);
    e->n = store;

    for (int r = 0; r < store; ++r) {
        const int j       = order[r];
        const int legal_i = qidx[j];

        const float rel =
            std::log(std::max(qp[j], 1e-9f) * static_cast<float>(nq));

        e->mv[r]        = legals[legal_i].move();
        e->rank[r]      = static_cast<int8_t>(std::min(r, 127));
        e->bonus[r]     = static_cast<int16_t>(shapeOrderBonus(rel, r, sharp));
        e->lmr_delta[r] = static_cast<int8_t>(shapeLMRDelta(rel, r, nq, sharp));
    }

    return e;
}