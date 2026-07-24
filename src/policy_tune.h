#pragma once

#include <string>

// ============================================================================
// Policy feature strengths for SPSA / OpenBench tuning
//
// Convention:
//   0   = feature disabled
//   5+  = feature enabled
//   100 = full strength
//
// In search code, use:
//   float w = policyWeight(g_pt.internal_policy_order);
//   if (w > 0.0f) { ... effect *= w ... }
// ============================================================================

struct PolicyTunables {
    // ------------------------------------------------------------------
    // Root policy features
    // ------------------------------------------------------------------
    int root_policy_order       = 100;
    int root_policy_lmr         = 100;
    int root_policy_verify      = 100;
    int root_tie_break          = 15;
    int root_policy_tail        = 30;
    int root_candidate          = 25;

    // ------------------------------------------------------------------
    // Time management / uncertainty
    // ------------------------------------------------------------------
    int policy_tm_continuous    = 50;
    int policy_entropy_tm       = 30;
    int policy_stability_tm     = 20;
    int policy_disagree_time    = 30;
    int aspiration_entropy      = 15;

    // ------------------------------------------------------------------
    // Internal policy ordering / reductions
    // ------------------------------------------------------------------
    int internal_policy_order   = 25;
    int internal_policy_lmr     = 20;
    int policy_top_protect_lmr  = 25;
    int policy_tail_reduce      = 20;
    int policy_iid              = 20;

    // ------------------------------------------------------------------
    // Pruning
    // ------------------------------------------------------------------
    int policy_lmp              = 15;
    int policy_whitelist        = 15;
    int policy_cumulative_mass  = 10;
    int policy_futility         = 15;
    int policy_see              = 10;
    int policy_history_prune    = 15;
    int policy_bad_capture      = 10;
    int policy_protect          = 25;
    int policy_hard_prune       = 0;

    // ------------------------------------------------------------------
    // Extensions / verification
    // ------------------------------------------------------------------
    int policy_extension        = 15;
    int policy_negative_extension = 5;
    int policy_se_gate          = 10;
    int policy_disagree_verify  = 20;
    int policy_probcut          = 5;

    // ------------------------------------------------------------------
    // Tactical / qsearch
    // ------------------------------------------------------------------
    int policy_sacrifice        = 10;
    int qsearch_policy_order    = 10;
    int qsearch_policy_skip     = 5;
    int qsearch_policy_quiet    = 0;
};

extern PolicyTunables g_pt;

void initPolicyParams();

bool setPolicyParam(const std::string& name, const std::string& value);

void printPolicyOptions();

inline float policyWeight(int strength) {
    if (strength < 5) return 0.0f;
    if (strength > 100) strength = 100;
    return static_cast<float>(strength) / 100.0f;
}