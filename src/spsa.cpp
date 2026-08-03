#include "spsa.h"
#include <iostream>

// ---- Tier 1 defaults ----
double spsa_lmr_base             = 0.75;
double spsa_lmr_divisor          = 2.25;
int    spsa_nmp_base             = 3;
int    spsa_nmp_divisor          = 3;
int    spsa_rfp_improving        = 70;
int    spsa_rfp_no_improving     = 95;
int    spsa_rfp_max_depth        = 7;
int    spsa_futility_base        = 100;
int    spsa_futility_per_depth   = 80;
int    spsa_see_noisy            = 50;
int    spsa_see_quiet            = 50;
int    spsa_asp_delta            = 25;

// ---- Tier 2 defaults ----
int    spsa_razor_d1             = 300;
int    spsa_razor_d2             = 500;
int    spsa_razor_d3             = 700;
int    spsa_hist_prune           = 2000;
int    spsa_lmp_base             = 3;
double spsa_lmp_depth_coeff      = 1.0;
int    spsa_se_margin            = 2;
int    spsa_se_double_bias       = 55;
int    spsa_se_triple_bias       = 120;
int    spsa_probcut_beta_margin  = 150;
int    spsa_probcut_see_thresh   = 100;
int    spsa_sprobcut_beta_margin = 350;
int    spsa_lmr_hist_divisor     = 4096;
int    spsa_hist_bonus           = 32;
int    spsa_pv_hist_bonus        = 8;

// ---- Tier 3 defaults ----
int    spsa_policy_bonus_scale   = 1600;
double spsa_policy_rank_scale    = 1.0;
double spsa_policy_tm_disagree   = 1.35;
double spsa_policy_tm_agree      = 0.88;
float  spsa_policy_sharp_thresh  = 0.25f;

// CSV: name, start, min, max, c, r
// c = perturbation size, r = learning-rate base. Tune per OpenBench convention.
void printSPSA() {
    std::cout << "lmr_base,"             << spsa_lmr_base             << ",0.40,1.50,0.05,0.5\n";
    std::cout << "lmr_divisor,"          << spsa_lmr_divisor          << ",1.50,3.50,0.10,0.5\n";
    std::cout << "nmp_base,"             << spsa_nmp_base             << ",2,5,1,0.5\n";
    std::cout << "nmp_divisor,"          << spsa_nmp_divisor          << ",2,5,1,0.5\n";
    std::cout << "rfp_improving,"        << spsa_rfp_improving        << ",45,110,5,0.5\n";
    std::cout << "rfp_no_improving,"     << spsa_rfp_no_improving     << ",60,140,5,0.5\n";
    std::cout << "rfp_max_depth,"        << spsa_rfp_max_depth        << ",5,10,1,0.5\n";
    std::cout << "futility_base,"        << spsa_futility_base        << ",50,200,10,0.5\n";
    std::cout << "futility_per_depth,"   << spsa_futility_per_depth   << ",40,140,10,0.5\n";
    std::cout << "see_noisy,"            << spsa_see_noisy            << ",20,100,5,0.5\n";
    std::cout << "see_quiet,"            << spsa_see_quiet            << ",20,100,5,0.5\n";
    std::cout << "asp_delta,"            << spsa_asp_delta            << ",10,50,3,0.5\n";
    std::cout << "razor_d1,"             << spsa_razor_d1             << ",150,500,25,0.5\n";
    std::cout << "razor_d2,"             << spsa_razor_d2             << ",300,750,25,0.5\n";
    std::cout << "razor_d3,"             << spsa_razor_d3             << ",450,1000,25,0.5\n";
    std::cout << "hist_prune,"           << spsa_hist_prune           << ",1000,4000,200,0.5\n";
    std::cout << "lmp_base,"             << spsa_lmp_base             << ",1,6,1,0.5\n";
    std::cout << "lmp_depth_coeff,"      << spsa_lmp_depth_coeff      << ",0.50,2.00,0.10,0.5\n";
    std::cout << "se_margin,"            << spsa_se_margin            << ",1,4,1,0.5\n";
    std::cout << "se_double_bias,"       << spsa_se_double_bias       << ",20,120,10,0.5\n";
    std::cout << "se_triple_bias,"       << spsa_se_triple_bias       << ",50,250,15,0.5\n";
    std::cout << "probcut_beta_margin,"  << spsa_probcut_beta_margin  << ",80,300,15,0.5\n";
    std::cout << "probcut_see_thresh,"   << spsa_probcut_see_thresh   << ",50,200,10,0.5\n";
    std::cout << "sprobcut_beta_margin," << spsa_sprobcut_beta_margin << ",200,550,25,0.5\n";
    std::cout << "lmr_hist_divisor,"     << spsa_lmr_hist_divisor     << ",2048,8192,512,0.5\n";
    std::cout << "hist_bonus,"           << spsa_hist_bonus           << ",16,64,4,0.5\n";
    std::cout << "pv_hist_bonus,"        << spsa_pv_hist_bonus        << ",4,20,2,0.5\n";
    std::cout << "policy_bonus_scale,"   << spsa_policy_bonus_scale   << ",800,3200,200,0.5\n";
    std::cout << "policy_rank_scale,"    << spsa_policy_rank_scale    << ",0.50,2.00,0.10,0.5\n";
    std::cout << "policy_tm_disagree,"   << spsa_policy_tm_disagree   << ",1.10,1.80,0.05,0.5\n";
    std::cout << "policy_tm_agree,"      << spsa_policy_tm_agree      << ",0.70,1.00,0.03,0.5\n";
    std::cout << "policy_sharp_thresh,"  << spsa_policy_sharp_thresh  << ",0.10,0.50,0.05,0.5\n";
    std::cout.flush();
}