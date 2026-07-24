#include "policy_tune.h"

#include <iostream>
#include <string>

PolicyTunables g_pt;

struct PolicyParamDesc {
    const char* name;
    int PolicyTunables::* field;
    int def;
    int min;
    int max;
};

// ============================================================================
// All tunable policy options
//
// These names are what OpenBench / cutechess will use:
//
//   setoption name InternalPolicyOrder value 35
//   setoption name PolicyTailReduce value 25
//
// ============================================================================

static const PolicyParamDesc kPolicyParams[] = {
    // Root
    { "RootPolicyOrder",          &PolicyTunables::root_policy_order,          100, 0, 100 },
    { "RootPolicyLMR",            &PolicyTunables::root_policy_lmr,            100, 0, 100 },
    { "RootPolicyVerify",         &PolicyTunables::root_policy_verify,         100, 0, 100 },
    { "RootTieBreak",             &PolicyTunables::root_tie_break,              15, 0, 100 },
    { "RootPolicyTail",           &PolicyTunables::root_policy_tail,            30, 0, 100 },
    { "RootCandidate",            &PolicyTunables::root_candidate,              25, 0, 100 },

    // TM / uncertainty
    { "PolicyTMContinuous",       &PolicyTunables::policy_tm_continuous,        50, 0, 100 },
    { "PolicyEntropyTM",          &PolicyTunables::policy_entropy_tm,           30, 0, 100 },
    { "PolicyStabilityTM",        &PolicyTunables::policy_stability_tm,         20, 0, 100 },
    { "PolicyDisagreeTime",       &PolicyTunables::policy_disagree_time,        30, 0, 100 },
    { "AspirationEntropy",        &PolicyTunables::aspiration_entropy,          15, 0, 100 },

    // Internal ordering / reductions
    { "InternalPolicyOrder",      &PolicyTunables::internal_policy_order,       25, 0, 100 },
    { "InternalPolicyLMR",        &PolicyTunables::internal_policy_lmr,         20, 0, 100 },
    { "PolicyTopProtectLMR",      &PolicyTunables::policy_top_protect_lmr,      25, 0, 100 },
    { "PolicyTailReduce",         &PolicyTunables::policy_tail_reduce,          20, 0, 100 },
    { "PolicyIID",                &PolicyTunables::policy_iid,                  20, 0, 100 },

    // Pruning
    { "PolicyLMP",                &PolicyTunables::policy_lmp,                  15, 0, 100 },
    { "PolicyWhitelist",          &PolicyTunables::policy_whitelist,            15, 0, 100 },
    { "PolicyCumulativeMass",     &PolicyTunables::policy_cumulative_mass,      10, 0, 100 },
    { "PolicyFutility",           &PolicyTunables::policy_futility,             15, 0, 100 },
    { "PolicySEE",                &PolicyTunables::policy_see,                  10, 0, 100 },
    { "PolicyHistoryPrune",       &PolicyTunables::policy_history_prune,        15, 0, 100 },
    { "PolicyBadCapture",         &PolicyTunables::policy_bad_capture,          10, 0, 100 },
    { "PolicyProtect",            &PolicyTunables::policy_protect,              25, 0, 100 },
    { "PolicyHardPrune",          &PolicyTunables::policy_hard_prune,            0, 0, 100 },

    // Extensions / verification
    { "PolicyExtension",          &PolicyTunables::policy_extension,            15, 0, 100 },
    { "PolicyNegativeExtension",  &PolicyTunables::policy_negative_extension,    5, 0, 100 },
    { "PolicySEGate",             &PolicyTunables::policy_se_gate,              10, 0, 100 },
    { "PolicyDisagreeVerify",     &PolicyTunables::policy_disagree_verify,      20, 0, 100 },
    { "PolicyProbCut",            &PolicyTunables::policy_probcut,               5, 0, 100 },

    // Tactical / qsearch
    { "PolicySacrifice",          &PolicyTunables::policy_sacrifice,            10, 0, 100 },
    { "QSearchPolicyOrder",       &PolicyTunables::qsearch_policy_order,        10, 0, 100 },
    { "QSearchPolicySkip",        &PolicyTunables::qsearch_policy_skip,          5, 0, 100 },
    { "QSearchPolicyQuiet",       &PolicyTunables::qsearch_policy_quiet,         0, 0, 100 },
};

static int clampInt(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int parseIntOr(const std::string& s, int def) {
    try {
        return std::stoi(s);
    } catch (...) {
        return def;
    }
}

void initPolicyParams() {
    for (const auto& p : kPolicyParams) {
        g_pt.*(p.field) = p.def;
    }
}

bool setPolicyParam(const std::string& name, const std::string& value) {
    for (const auto& p : kPolicyParams) {
        if (name == p.name) {
            int current = g_pt.*(p.field);
            int v = parseIntOr(value, current);
            v = clampInt(v, p.min, p.max);
            g_pt.*(p.field) = v;
            return true;
        }
    }

    return false;
}

void printPolicyOptions() {
    for (const auto& p : kPolicyParams) {
        std::cout << "option name " << p.name
                  << " type spin"
                  << " default " << p.def
                  << " min " << p.min
                  << " max " << p.max
                  << std::endl;
    }
}