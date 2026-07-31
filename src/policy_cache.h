#pragma once
// ============================================================================
// policy_cache.h
// One-pass internal policy evaluation + per-thread Zobrist cache.
// Shared by the pseudo-TT injection (search.cpp) and the MovePicker ordering.
// ============================================================================
#include "policy.h"          // g_policy, policyQuietLocal, PolicyNet
#include "chess.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

// ----------------------------------------------------------------------------
// Everything derived from a SINGLE forward pass at one position.
// ----------------------------------------------------------------------------
struct PolicyNodeEval {
    uint64_t      key   = 0;
    bool          valid = false;

    // all-move softmax (used for pseudo-TT top move)
    chess::Move   top_any{};
    float         top_prob_any    = 0.0f;
    float         norm_entropy_any = 1.0f;

    // quiet-only model (used for ordering)
    float         quiet_sharpness = 0.0f;
    int           nq              = 0;

    // Top quiet moves stored in rank order. We only ever need the top ranks
    // (bonus is 0 beyond rank 10), so 64 slots is more than enough.
    chess::Move   quiet_moves[64];
    int8_t        quiet_rank[64];   // rank within quiets, 0 == best

    int quietRankOf(const chess::Move& m) const {
        for (int i = 0; i < nq; ++i)
            if (quiet_moves[i] == m) return quiet_rank[i];
        return -1;
    }
};

// ----------------------------------------------------------------------------
// Compute a PolicyNodeEval with exactly ONE network forward pass.
// Mirrors computeRootPolicy()'s softmax + sharpness formula for consistency.
// ----------------------------------------------------------------------------
inline bool computeNodePolicy(const chess::Board& board, PolicyNodeEval& out) {
    out.valid           = false;
    out.top_any         = chess::Move();
    out.top_prob_any    = 0.0f;
    out.norm_entropy_any = 1.0f;
    out.quiet_sharpness = 0.0f;
    out.nq              = 0;

    if (!g_policy.loaded) return false;

    chess::Movelist legals;
    chess::movegen::legalmoves(legals, board);
    const int n = static_cast<int>(legals.size());
    if (n < 1 || n > 256) return false;

    float logits[256];
    if (!g_policy.logitsLegalMoves(board, legals, logits)) return false;  // <-- the ONE pass

    // ---- all-move softmax ----
    float mx = -1e30f;
    for (int i = 0; i < n; ++i) mx = std::max(mx, logits[i]);

    float probs[256];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { probs[i] = std::exp(logits[i] - mx); sum += probs[i]; }
    if (sum <= 0.0f) sum = 1.0f;

    int   best_i = 0;
    float best_p = -1.0f;
    double ent   = 0.0;
    for (int i = 0; i < n; ++i) {
        const float p = probs[i] / sum;
        if (p > best_p) { best_p = p; best_i = i; }
        if (p > 1e-12f) ent -= double(p) * std::log(double(p));
    }
    out.top_any      = legals[best_i];
    out.top_prob_any = best_p;
    out.norm_entropy_any =
        (n > 1) ? static_cast<float>(ent / std::log(static_cast<float>(n))) : 0.0f;

    // ---- quiet-only softmax + ranks + sharpness ----
    int   qidx[256];
    int   nq   = 0;
    float maxq = -1e30f;
    for (int i = 0; i < n; ++i) {
        if (!policyQuietLocal(board, legals[i])) continue;
        qidx[nq++] = i;
        maxq = std::max(maxq, logits[i]);
    }
    out.nq = nq;

    if (nq > 0) {
        float qp[256];
        float sq = 0.0f;
        for (int j = 0; j < nq; ++j) { qp[j] = std::exp(logits[qidx[j]] - maxq); sq += qp[j]; }
        if (sq <= 0.0f) sq = 1.0f;
        for (int j = 0; j < nq; ++j) qp[j] /= sq;

        int order[256];
        for (int j = 0; j < nq; ++j) order[j] = j;
        std::sort(order, order + nq, [&](int a, int b) { return qp[a] > qp[b]; });

        const float top_quiet_prob = qp[order[0]];
        int stored = 0;
        for (int r = 0; r < nq && stored < 64; ++r) {
            const int li = qidx[order[r]];
            out.quiet_moves[stored] = legals[li];
            out.quiet_rank[stored]  = static_cast<int8_t>(r);
            ++stored;
        }

        double qent = 0.0;
        for (int j = 0; j < nq; ++j)
            if (qp[j] > 1e-12f) qent -= double(qp[j]) * std::log(double(qp[j]));

        const float normq =
            (nq > 1) ? static_cast<float>(qent / std::log(static_cast<float>(nq))) : 0.0f;

        // identical to computeRootPolicy()
        float sharp = std::clamp((0.90f - normq) / 0.35f, 0.0f, 1.0f);
        if (top_quiet_prob < 0.12f) sharp *= 0.5f;
        out.quiet_sharpness = sharp;
    }

    out.valid = true;
    return true;
}

// ----------------------------------------------------------------------------
// Per-thread, direct-mapped, lock-free cache.
// Key validation makes stale entries self-correcting (no clear strictly needed).
// ----------------------------------------------------------------------------
struct PolicyCache {
    static constexpr int SIZE = 2048;          // power of two; ~450 KB / thread
    PolicyNodeEval entries[SIZE];

    PolicyNodeEval* probe(uint64_t key) {
        PolicyNodeEval& e = entries[key & (SIZE - 1)];
        return (e.valid && e.key == key) ? &e : nullptr;
    }
    void clear() {
        for (auto& e : entries) { e.valid = false; e.key = 0; }
    }
};

inline PolicyCache& policyCache() {
    thread_local PolicyCache cache;            // one instance per thread
    return cache;
}

// The single entry point both consumers use.
inline PolicyNodeEval* getOrComputeNodePolicy(const chess::Board& board, uint64_t key) {
    if (!g_policy.loaded) return nullptr;
    PolicyCache& c = policyCache();
    if (PolicyNodeEval* hit = c.probe(key)) return hit;

    PolicyNodeEval& slot = c.entries[key & (PolicyCache::SIZE - 1)];
    if (computeNodePolicy(board, slot)) {
        slot.key = key;                        // valid already set by computeNodePolicy
        return &slot;
    }
    slot.valid = false;
    return nullptr;
}

inline void clearPolicyCache() { policyCache().clear(); }