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

// Change this to 1024 if your embedded policy blob is the 1024 HL net.
// For the old 128 blob, leave it at 128.
#ifndef KOCIOLEK_POLICY_HL
#define KOCIOLEK_POLICY_HL 128
#endif

constexpr int POLICY_HL          = KOCIOLEK_POLICY_HL;
constexpr int POLICY_HL_PAIR     = POLICY_HL / 2;

constexpr int POLICY_MAX_ACTIVE  = 32;
constexpr int POLICY_QA          = 128;
constexpr int POLICY_PROMOS      = 4 * 22;           // 88
constexpr int POLICY_SEE_TH      = -108;

// Legacy constants kept for compatibility/debug output.
constexpr int POLICY_ROOT_LMR_MIN_DEPTH = 3;
constexpr int POLICY_ROOT_LMR_TOP       = 3;

// Legacy.
constexpr int POLICY_QUIET_WEIGHT  = 1024;
constexpr int POLICY_MIN_DEPTH     = 6;

// ============================================================================
// Policy-disagreement time management (root only, zero NPS tax in-tree)
// ============================================================================

constexpr int    POLICY_TM_MIN_DEPTH    = 6;
constexpr float  POLICY_TM_AGREE_CONF   = 0.35f;
constexpr float  POLICY_TM_UNCERTAIN    = 0.18f;
constexpr double POLICY_TM_DISAGREE     = 1.35;
constexpr double POLICY_TM_UNCERTAIN_S  = 1.25;
constexpr double POLICY_TM_AGREE_S      = 0.88;

constexpr float POLICY_REL_NONE = -1000.0f;

// ============================================================================
// RootPolicy
// ============================================================================

struct RootPolicy {
    bool ok = false;

    int nlegal = 0;
    int nq = 0;

    chess::Movelist legals;

    // Per legal move:
    // rel[i] = log(quiet_prob[i] * nq) for quiets, POLICY_REL_NONE otherwise.
    float rel[256];

    // Quiet-only probability for quiets, 0 for non-quiets.
    float quiet_prob[256];

    // All-legal-move softmax probability, used for TM.
    float prob_any[256];

    // Quiet rank:
    // -1 for non-quiet,
    // 0..nq-1 for quiets, 0 = best quiet.
    int quiet_rank[256];

    // Best legal move according to all-move softmax.
    chess::Move top_any;
    float top_prob_any = 0.0f;
    float entropy_any = 0.0f;
    float norm_entropy_any = 1.0f;

    // Best quiet according to quiet-only softmax.
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
// PolicyNet
// ============================================================================

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

// ============================================================================
// Root policy computation
// ============================================================================

inline bool policyQuietLocal(const chess::Board& board, const chess::Move& m) {
    if (m.typeOf() == chess::Move::PROMOTION) return false;
    if (m.typeOf() == chess::Move::ENPASSANT) return false;
    if (board.at(m.to()) != chess::Piece::NONE) return false;
    return true; // includes castling
}

inline bool computeRootPolicy(const chess::Board& board, RootPolicy& rp) {
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

    if (!g_policy.loaded) {
        return false;
    }

    chess::movegen::legalmoves(rp.legals, board);
    rp.nlegal = static_cast<int>(rp.legals.size());

    if (rp.nlegal < 1 || rp.nlegal > 256) {
        return false;
    }

    float logits[256];

    if (!g_policy.logitsLegalMoves(board, rp.legals, logits)) {
        return false;
    }

    // ------------------------------------------------------------------------
    // All-legal-move softmax, used for time management.
    // ------------------------------------------------------------------------

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

    if (sum_all <= 0.0f) {
        sum_all = 1.0f;
    }

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

    // ------------------------------------------------------------------------
    // Quiet-only softmax, used for ordering / LMR / protection.
    // ------------------------------------------------------------------------

    int qidx[256];
    int nq = 0;

    float max_q = -1e30f;

    for (int i = 0; i < rp.nlegal; ++i) {
        if (!policyQuietLocal(board, rp.legals[i])) {
            continue;
        }

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

        if (sum_q <= 0.0f) {
            sum_q = 1.0f;
        }

        for (int j = 0; j < nq; ++j) {
            qp[j] /= sum_q;

            const int legal_i = qidx[j];

            rp.quiet_prob[legal_i] = qp[j];
            rp.rel[legal_i] =
                std::log(std::max(qp[j], 1e-9f) * static_cast<float>(nq));
        }

        int order[256];
        for (int j = 0; j < nq; ++j) {
            order[j] = j;
        }

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
            norm_qent =
                static_cast<float>(qent / std::log(static_cast<float>(nq)));
        }

        rp.quiet_norm_entropy = norm_qent;

        float sharpness =
            std::clamp((0.90f - norm_qent) / 0.35f, 0.0f, 1.0f);

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