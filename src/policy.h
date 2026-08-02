#pragma once
#include "chess.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================================
// Matches monty/bullet policy trainer (inputs.rs + model.rs)
// ============================================================================
constexpr int POLICY_PLANE       = 768;
constexpr int POLICY_INPUT_SIZE  = POLICY_PLANE * 4; // 3072

#ifndef KOCIOLEK_POLICY_HL
#define KOCIOLEK_POLICY_HL 2048
#endif
#ifndef POLICY_SMALL_HL
#define POLICY_SMALL_HL 64
#endif

constexpr int POLICY_HL          = KOCIOLEK_POLICY_HL;
constexpr int POLICY_HL_PAIR     = POLICY_HL / 2;
constexpr int POLICY_MAX_ACTIVE  = 32;
constexpr int POLICY_QA          = 128;
constexpr int POLICY_PROMOS      = 4 * 22;           // 88
constexpr int POLICY_SEE_TH      = -108;

constexpr int POLICY_ROOT_LMR_MIN_DEPTH = 3;
constexpr int POLICY_ROOT_LMR_TOP       = 3;
constexpr int POLICY_QUIET_WEIGHT  = 1024;
constexpr int POLICY_MIN_DEPTH     = 6;

// ============================================================================
// Policy-disagreement time management
// ============================================================================
constexpr int    POLICY_TM_MIN_DEPTH    = 6;
constexpr float  POLICY_TM_AGREE_CONF   = 0.35f;
constexpr float  POLICY_TM_UNCERTAIN    = 0.18f;
constexpr double POLICY_TM_DISAGREE     = 1.35;
constexpr double POLICY_TM_UNCERTAIN_S  = 1.25;
constexpr double POLICY_TM_AGREE_S      = 0.88;
constexpr float  POLICY_TM_ENTROPY_GATE = 0.80f;

constexpr float POLICY_REL_NONE = -1000.0f;

// ============================================================================
// RootPolicy
// ============================================================================
struct RootPolicy {
    bool ok = false;
    int nlegal = 0;
    int nq = 0;

    chess::Movelist legals;
    float rel[256];
    float quiet_prob[256];
    float prob_any[256];
    int   quiet_rank[256];

    chess::Move top_any;
    float top_prob_any = 0.0f;
    float entropy_any = 0.0f;
    float norm_entropy_any = 1.0f;

    chess::Move top_quiet;
    float top_quiet_prob = 0.0f;
    float quiet_entropy = 0.0f;
    float quiet_norm_entropy = 1.0f;
    float quiet_sharpness = 0.0f;

    int find(const chess::Move& m) const {
        for (int i = 0; i < nlegal && i < 256; ++i) {
            if (legals[i] == m) return i;
        }
        return -1;
    }

    bool is_quiet_move(const chess::Move& m) const {
        const int i = find(m);
        return i >= 0 && quiet_rank[i] >= 0;
    }

    float rel_of(const chess::Move& m) const {
        const int i = find(m);
        return (i >= 0) ? rel[i] : POLICY_REL_NONE;
    }

    bool protected_quiet(const chess::Move& m) const {
        if (!ok) return false;
        if (quiet_sharpness < 0.25f) return false;
        const int i = find(m);
        if (i < 0) return false;
        const int r = quiet_rank[i];
        if (r < 0) return false;
        if (r <= 1) return true;
        return rel[i] > 0.0f;
    }
};

// ============================================================================
// PolicyNetT
// ============================================================================
template <int Hidden_>
struct PolicyNetT {
    static_assert(Hidden_ > 0 && (Hidden_ % 2) == 0,
                  "Policy hidden layer size must be positive and even");
    static constexpr int HL       = Hidden_;
    static constexpr int HL_PAIR  = Hidden_ / 2;

    bool loaded = false;
    int from_to   = 0;
    int num_moves = 0;

    std::vector<float> l0w;
    std::vector<float> l0b;
    std::vector<float> l1w;
    std::vector<float> l1b;
    std::vector<float> l2w;
    std::vector<float> l2b;

    uint64_t destinations[64][6]{};
    int      offsets[6][65]{};

    PolicyNetT();
    void clear();
    bool loadFromMemory(const std::uint8_t* data,
                        std::size_t size,
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

    void collectFeatures(const chess::Board& board,
                         int* feats,
                         int& nfeats) const;
    int mapMoveToIndex(const chess::Board& board,
                       const chess::Move& m) const;

    void debugPosition(const chess::Board& board, int topN = 16) const;
    void debugMove(const chess::Board& board, const chess::Move& m) const;

    static int      stmKingIndex(const chess::Board& board);
    static int      flipMask(const chess::Board& board);
    static uint64_t attacksBySide(const chess::Board& board, chess::Color side);
};

using PolicyNet      = PolicyNetT<POLICY_HL>;
using PolicyNetSmall = PolicyNetT<POLICY_SMALL_HL>;

extern PolicyNet      g_policy;
extern PolicyNetSmall g_policy_small;

// ============================================================================
// Root policy computation
// ============================================================================
inline bool policyQuietLocal(const chess::Board& board, const chess::Move& m) {
    if (m.typeOf() == chess::Move::PROMOTION) return false;
    if (m.typeOf() == chess::Move::ENPASSANT) return false;
    if (board.at(m.to()) != chess::Piece::NONE) return false;
    return true;
}

template <int HL>
inline bool computeRootPolicyForNet(const PolicyNetT<HL>& net,
                                    const chess::Board& board,
                                    RootPolicy& rp) {
    rp.ok = false;
    rp.nlegal = 0;
    rp.nq = 0;
    rp.legals.clear();
    rp.top_any = chess::Move();
    rp.top_prob_any = 0.0f;
    rp.entropy_any = 0.0f;
    rp.norm_entropy_any = 1.0f;
    rp.top_quiet = chess::Move();
    rp.top_quiet_prob = 0.0f;
    rp.quiet_entropy = 0.0f;
    rp.quiet_norm_entropy = 1.0f;
    rp.quiet_sharpness = 0.0f;

    for (int i = 0; i < 256; ++i) {
        rp.rel[i] = POLICY_REL_NONE;
        rp.quiet_prob[i] = 0.0f;
        rp.prob_any[i] = 0.0f;
        rp.quiet_rank[i] = -1;
    }

    if (!net.loaded) {
        return false;
    }

    chess::movegen::legalmoves(rp.legals, board);
    rp.nlegal = static_cast<int>(rp.legals.size());
    if (rp.nlegal < 1 || rp.nlegal > 256) {
        return false;
    }

    float logits[256];
    if (!net.logitsLegalMoves(board, rp.legals, logits)) {
        return false;
    }

    // All-legal-move softmax
    float max_all = -1e30f;
    for (int i = 0; i < rp.nlegal; ++i) {
        max_all = std::max(max_all, logits[i]);
    }
    float probs_all[256];
    float sum_all = 0.0f;
    for (int i = 0; i < rp.nlegal; ++i) {
        probs_all[i] = std::exp(logits[i] - max_all);
        sum_all += probs_all[i];
    }
    if (sum_all <= 0.0f) sum_all = 1.0f;

    int best_any_i = 0;
    float best_any_p = -1.0f;
    double ent_any = 0.0;
    for (int i = 0; i < rp.nlegal; ++i) {
        const float p = probs_all[i] / sum_all;
        rp.prob_any[i] = p;
        if (p > best_any_p) {
            best_any_p = p;
            best_any_i = i;
        }
        if (p > 1e-12f) {
            ent_any -= double(p) * std::log(double(p));
        }
    }
    rp.top_any = rp.legals[best_any_i];
    rp.top_prob_any = best_any_p;
    rp.entropy_any = static_cast<float>(ent_any);
    if (rp.nlegal > 1) {
        rp.norm_entropy_any =
            static_cast<float>(ent_any / std::log(static_cast<float>(rp.nlegal)));
    } else {
        rp.norm_entropy_any = 0.0f;
    }

    // Quiet-only softmax
    int qidx[256];
    int nq = 0;
    float max_q = -1e30f;
    for (int i = 0; i < rp.nlegal; ++i) {
        if (!policyQuietLocal(board, rp.legals[i])) continue;
        qidx[nq++] = i;
        max_q = std::max(max_q, logits[i]);
    }
    rp.nq = nq;

    if (nq > 0) {
        float sum_q = 0.0f;
        float qp[256];
        for (int j = 0; j < nq; ++j) {
            qp[j] = std::exp(logits[qidx[j]] - max_q);
            sum_q += qp[j];
        }
        if (sum_q <= 0.0f) sum_q = 1.0f;

        for (int j = 0; j < nq; ++j) {
            qp[j] /= sum_q;
            const int legal_i = qidx[j];
            rp.quiet_prob[legal_i] = qp[j];
            rp.rel[legal_i] =
                std::log(std::max(qp[j], 1e-9f) * static_cast<float>(nq));
        }

        int order[256];
        for (int j = 0; j < nq; ++j) order[j] = j;
        std::sort(order, order + nq, [&](int a, int b) {
            return qp[a] > qp[b];
        });
        for (int r = 0; r < nq; ++r) {
            const int legal_i = qidx[order[r]];
            rp.quiet_rank[legal_i] = r;
        }

        const int top_legal_i = qidx[order[0]];
        rp.top_quiet = rp.legals[top_legal_i];
        rp.top_quiet_prob = rp.quiet_prob[top_legal_i];

        double qent = 0.0;
        for (int j = 0; j < nq; ++j) {
            if (qp[j] > 1e-12f) {
                qent -= double(qp[j]) * std::log(double(qp[j]));
            }
        }
        rp.quiet_entropy = static_cast<float>(qent);
        float norm_qent = 0.0f;
        if (nq > 1) {
            norm_qent = static_cast<float>(qent / std::log(static_cast<float>(nq)));
        }
        rp.quiet_norm_entropy = norm_qent;

        float sharpness = std::clamp((0.90f - norm_qent) / 0.35f, 0.0f, 1.0f);
        if (rp.top_quiet_prob < 0.12f) {
            sharpness *= 0.5f;
        }
        rp.quiet_sharpness = sharpness;
    } else {
        rp.quiet_sharpness = 0.0f;
    }

    rp.ok = true;
    return true;
}

inline bool computeRootPolicy(const chess::Board& board, RootPolicy& rp) {
    return computeRootPolicyForNet(g_policy, board, rp);
}

inline bool computeSmallRootPolicy(const chess::Board& board, RootPolicy& rp) {
    return computeRootPolicyForNet(g_policy_small, board, rp);
}

// ============================================================================
// In-tree node policy (small net only, computed per node)
// Cost: ~5k MACs (cheaper than one NNUE eval). Gate externally by depth.
// ============================================================================

struct NodePolicy {
    bool valid = false;
    int nlegal = 0;
    int nq = 0;
    float rel[64];            // log-relative quiet prob per legal-move index
    int   quiet_rank[64];     // rank among quiets (-1 = not quiet)
    float sharpness = 0.0f;
    chess::Move legals[64];   // compact storage, no heap

    int find(const chess::Move& m) const {
        for (int i = 0; i < nlegal; ++i)
            if (legals[i] == m) return i;
        return -1;
    }
};

inline bool computeNodePolicy(const chess::Board& board, NodePolicy& np) {
    np.valid = false;
    np.nlegal = 0;
    np.nq = 0;
    np.sharpness = 0.0f;

    if (!g_policy_small.loaded) return false;

    chess::Movelist ml;
    chess::movegen::legalmoves(ml, board);
    np.nlegal = static_cast<int>(ml.size());
    if (np.nlegal < 2 || np.nlegal > 64) return false;

    for (int i = 0; i < np.nlegal; ++i) {
        np.legals[i] = ml[i];
        np.rel[i] = POLICY_REL_NONE;
        np.quiet_rank[i] = -1;
    }

    float logits[64];
    if (!g_policy_small.logitsLegalMoves(board, ml, logits)) return false;

    // Quiet-only softmax
    int qidx[64];
    int nq = 0;
    float max_q = -1e30f;
    for (int i = 0; i < np.nlegal; ++i) {
        if (policyQuietLocal(board, ml[i])) {
            qidx[nq++] = i;
            if (logits[i] > max_q) max_q = logits[i];
        }
    }
    np.nq = nq;
    if (nq < 2) return false;

    float sum_q = 0.0f;
    float qp[64];
    for (int j = 0; j < nq; ++j) {
        qp[j] = std::exp(logits[qidx[j]] - max_q);
        sum_q += qp[j];
    }
    if (sum_q <= 0.0f) return false;

    for (int j = 0; j < nq; ++j) {
        qp[j] /= sum_q;
        np.rel[qidx[j]] = std::log(std::max(qp[j], 1e-9f) * static_cast<float>(nq));
    }

    // Rank quiets
    int order[64];
    for (int j = 0; j < nq; ++j) order[j] = j;
    std::sort(order, order + nq, [&](int a, int b) { return qp[a] > qp[b]; });
    for (int r = 0; r < nq; ++r)
        np.quiet_rank[qidx[order[r]]] = r;

    // Sharpness
    double qent = 0.0;
    for (int j = 0; j < nq; ++j)
        if (qp[j] > 1e-12f) qent -= double(qp[j]) * std::log(double(qp[j]));
    float norm_ent = (nq > 1) ? static_cast<float>(qent / std::log(static_cast<float>(nq))) : 0.0f;
    np.sharpness = std::clamp((0.90f - norm_ent) / 0.35f, 0.0f, 1.0f);

    np.valid = true;
    return true;
}