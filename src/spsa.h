#pragma once

// ============================================================================
// Tier 1 – core pruning / reductions
// ============================================================================
extern double spsa_lmr_base;
extern double spsa_lmr_divisor;
extern int    spsa_nmp_base;
extern int    spsa_nmp_divisor;
extern int    spsa_rfp_improving;
extern int    spsa_rfp_no_improving;
extern int    spsa_rfp_max_depth;
extern int    spsa_futility_base;
extern int    spsa_futility_per_depth;
extern int    spsa_see_noisy;
extern int    spsa_see_quiet;
extern int    spsa_asp_delta;

// ============================================================================
// Tier 2 – secondary pruning / history
// ============================================================================
extern int    spsa_razor_d1;
extern int    spsa_razor_d2;
extern int    spsa_razor_d3;
extern int    spsa_hist_prune;
extern int    spsa_lmp_base;
extern double spsa_lmp_depth_coeff;
extern int    spsa_se_margin;
extern int    spsa_se_double_bias;
extern int    spsa_se_triple_bias;
extern int    spsa_probcut_beta_margin;
extern int    spsa_probcut_see_thresh;
extern int    spsa_sprobcut_beta_margin;
extern int    spsa_lmr_hist_divisor;
extern int    spsa_hist_bonus;
extern int    spsa_pv_hist_bonus;

// ============================================================================
// Tier 3 – policy integration
// ============================================================================
extern int    spsa_policy_bonus_scale;
extern double spsa_policy_rank_scale;
extern double spsa_policy_tm_disagree;
extern double spsa_policy_tm_agree;
extern float  spsa_policy_sharp_thresh;

// Dump all params as CSV for copy/paste into the OpenBench web UI
void printSPSA();