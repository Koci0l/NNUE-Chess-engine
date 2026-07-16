#include "policy_diag.h"

#include "policy.h"
#include "search.h"
#include "tt.h"
#include "history.h"
#include "nnue.h"
#include "types.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// Diverse FENs (openings + middlegames + some endgames)
// ============================================================================

static const char* kPolicyBenchFens[] = {
    // Openings
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
    "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",

    // Quiet middlegames
    "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQ - 0 7",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
    "r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9",
    "r2q1rk1/ppp2ppp/2n1bn2/2bpp3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 4 8",
    "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R w KQ - 0 8",
    "rnbq1rk1/ppp1bppp/4pn2/3p2B1/2PP4/2N2N2/PP2PPPP/R2QKB1R w KQ - 4 6",
    "r1bqr1k1/pp1n1ppp/2pbpn2/3p4/2PP4/1PN1PN2/P1Q2PPP/R1B1KB1R w KQ - 1 9",
    "r2q1rk1/1bp1bppp/p1np1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 0 11",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9",
    "rnbq1rk1/1p2bppp/p2p1n2/4p3/4P3/1NN1B3/PPP1BPPP/R2QK2R w KQ - 0 10",

    // More branching
    "r1bqk2r/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B1KB1R b KQkq - 2 7",
    "r2qk2r/pb1n1ppp/1ppbpn2/3p4/2PP4/1PN1PN2/P1Q1BPPP/R1B1K2R w KQkq - 2 9",
    "r1bq1rk1/1p2bppp/p1nppn2/8/3NPP2/2N1B3/PPP1B1PP/R2Q1RK1 b - - 0 10",
    "r2q1rk1/pp1nppbp/2p2np1/3p1b2/2PP4/1PN1PN2/PB1QBPPP/R3K2R w KQ - 3 10",
    "r1b1k2r/pp1n1ppp/2p1pn2/q2p2B1/1bPP4/2N1P3/PPQN1PPP/R3KB1R w KQkq - 4 8",
    "rnb1k2r/pp2qppp/4pn2/2pp4/1bPP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 0 8",
    "r1bqkb1r/pp3ppp/2nppn2/6B1/3NP3/2N5/PPP2PPP/R2QKB1R w KQkq - 0 7",
    "r2qkb1r/1b1n1ppp/p1p1pn2/1p6/3P4/2NBPN2/PP3PPP/R1BQK2R w KQkq - 0 9",
    "r1bqk2r/pp1pppbp/2n2np1/8/2PNP3/2N5/PP3PPP/R1BQKB1R w KQkq - 1 7",
    "rnbqk2r/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R b KQkq - 0 5",

    // Late middlegame
    "3r1rk1/1bq1bppp/p2ppn2/1p6/4P3/1NN1BP2/PPP3PP/2RQR1K1 w - - 0 16",
    "r4rk1/pp1qbppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 11",
    "2rq1rk1/pp1bppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/2KR3R w - - 6 12",
    "r2q1rk1/1p1bbppp/p1nppn2/8/3NPP2/2N1B3/PPP1B1PP/R2Q1RK1 w - - 0 12",
    "1r3rk1/2qbbppp/p2ppn2/1p6/3BPP2/2N3Q1/PPP1B1PP/3R1R1K w - - 0 16",
    "r1bq1rk1/pp1n1pbp/2pp1np1/4p3/2PPP3/2N1BN2/PP2BPPP/R2Q1RK1 w - - 0 9",
    "r2q1rk1/ppp1bppp/2n1bn2/3pp3/8/3P1NP1/PPP1PPBP/RNBQR1K1 w - - 0 9",
    "r1b2rk1/pp1nqppp/2p1p3/3n4/3P4/2NB1N2/PP3PPP/R2QK2R w KQ - 0 12",
    "2rr2k1/pp2bppp/1qn1pn2/3p1b2/1P1P1B2/P1N1PN2/4BPPP/2RQ1RK1 w - - 1 14",
    "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1B1P1N2/PPP2PPP/RNBQR1K1 w - - 0 9",
};

static constexpr int kNumBenchFens =
    static_cast<int>(sizeof(kPolicyBenchFens) / sizeof(kPolicyBenchFens[0]));

// ============================================================================
// Rank helpers
// ============================================================================

bool policyRankLegalMoves(const chess::Board& board,
                          const chess::Movelist& legal,
                          int* out_order,
                          float* out_logits,
                          int& n) {
    n = 0;
    if (!g_policy.loaded || legal.empty()) {
        return false;
    }

    n = static_cast<int>(legal.size());
    if (n > 256) {
        return false;
    }

    if (!g_policy.logitsLegalMoves(board, legal, out_logits)) {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        out_order[i] = i;
    }
    std::sort(out_order, out_order + n, [&](int a, int b) {
        return out_logits[a] > out_logits[b];
    });
    return true;
}

void runPolicyDebug(const chess::Board& board, int topN) {
    g_policy.debugPosition(board, topN);
}

// ============================================================================
// Soft state reset between FENs (avoid TT/history leaking between positions)
// ============================================================================

static void softResetSearchState() {
    advanceTTGeneration();
    // Optional harder isolation — uncomment if ranks look sticky across FENs:
    // clearTT();
    // g_butterflyHistory.clear();
    // g_killerMoves.clear();
    // g_counterMoves.clear();
    // g_captureHistory.clear();
    // g_contHist1ply.clear();
    // g_contHist2ply.clear();
    // g_correctionHistory.clear();
}

// ============================================================================
// One position
// ============================================================================

struct HitRow {
    std::string fen;
    std::string search_move;
    std::string policy_top1;
    std::string policy_top3;
    int rank = -1;   // 1-based; -1 if search move not in policy list
    int legal = 0;
    int search_score = 0;
    uint64_t nodes = 0;
    bool ok = false;
};

static HitRow evaluateOnePosition(const char* fen,
                                  int search_depth,
                                  int64_t search_nodes,
                                  ThreadInfo& thread) {
    HitRow row;
    row.fen = fen;

    chess::Board board;
    board.setFen(fen);
    thread.accumulatorStack.resetAccumulators(board);

    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);
    row.legal = static_cast<int>(legal.size());
    if (row.legal == 0) {
        return row;
    }

    float logits[256];
    int order[256];
    int n = 0;
    if (!policyRankLegalMoves(board, legal, order, logits, n)) {
        row.policy_top1 = "n/a";
        return row;
    }

    row.policy_top1 = chess::uci::moveToUci(legal[order[0]]);
    {
        std::string t3;
        for (int k = 0; k < std::min(3, n); ++k) {
            if (k) t3 += ' ';
            t3 += chess::uci::moveToUci(legal[order[k]]);
        }
        row.policy_top3 = t3;
    }

    softResetSearchState();
    thread.accumulatorStack.resetAccumulators(board);

    TimeManager tm;
    // Match your bench: no wall-clock budget; rely on depth / nodes
    tm.init(0, 0, 0, 0, 0);

    const int max_depth = (search_depth > 0) ? search_depth : 20;
    const int64_t node_limit = (search_depth > 0) ? 0 : search_nodes;

    int score = 0;
    uint64_t nodes = 0;

    const bool old_silent = g_silent;
    g_silent = true;

    chess::Move best = search(board, max_depth, thread, tm, node_limit, &score, &nodes);

    g_silent = old_silent;

    if (best == chess::Move()) {
        return row;
    }

    row.search_move = chess::uci::moveToUci(best);
    row.search_score = score;
    row.nodes = nodes;

    row.rank = -1;
    for (int r = 0; r < n; ++r) {
        if (legal[order[r]] == best) {
            row.rank = r + 1;
            break;
        }
    }

    row.ok = (row.rank > 0);
    return row;
}

// ============================================================================
// Public bench
// ============================================================================

void runPolicyHitBench(int search_depth, int64_t search_nodes, ThreadInfo& thread) {
    if (!g_policy.loaded) {
        std::cout << "info string Policy not loaded — cannot bench" << std::endl;
        std::cout.flush();
        return;
    }

    if (search_depth <= 0 && search_nodes <= 0) {
        search_depth = 10;
    }

    std::cout << "info string === POLICY HIT BENCH ===" << std::endl;
    std::cout << "info string positions=" << kNumBenchFens
              << " search_depth=" << search_depth
              << " search_nodes=" << search_nodes
              << " hl=" << POLICY_HL
              << " weight=" << POLICY_QUIET_WEIGHT
              << " see_th=" << POLICY_SEE_TH
              << std::endl;

    int top1 = 0, top3 = 0, top5 = 0, valid = 0, skipped = 0;
    double sum_recip_rank = 0.0;
    long long sum_rank = 0;
    int hist[22];
    std::memset(hist, 0, sizeof(hist));

    // Split: first 10 openings vs rest
    int open_n = 0, open_top1 = 0, open_top3 = 0;
    int mid_n = 0, mid_top1 = 0, mid_top3 = 0;

    for (int i = 0; i < kNumBenchFens; ++i) {
        HitRow row = evaluateOnePosition(kPolicyBenchFens[i], search_depth,
                                         search_nodes, thread);

        if (!row.ok) {
            ++skipped;
            std::cout << "info string [" << (i + 1) << "] SKIP"
                      << " legal=" << row.legal
                      << " pol1=" << row.policy_top1
                      << " search=" << row.search_move
                      << " fen " << row.fen
                      << std::endl;
            continue;
        }

        ++valid;
        if (row.rank == 1) ++top1;
        if (row.rank <= 3) ++top3;
        if (row.rank <= 5) ++top5;
        sum_rank += row.rank;
        sum_recip_rank += 1.0 / double(row.rank);

        const int bin = (row.rank <= 20) ? row.rank : 21;
        hist[bin]++;

        if (i < 10) {
            ++open_n;
            if (row.rank == 1) ++open_top1;
            if (row.rank <= 3) ++open_top3;
        } else {
            ++mid_n;
            if (row.rank == 1) ++mid_top1;
            if (row.rank <= 3) ++mid_top3;
        }

        std::cout << "info string [" << (i + 1) << "/" << kNumBenchFens << "]"
                  << " rank=" << row.rank << "/" << row.legal
                  << " search=" << row.search_move
                  << " pol1=" << row.policy_top1
                  << " top3=[" << row.policy_top3 << "]"
                  << " score=" << row.search_score
                  << " nodes=" << row.nodes
                  << std::endl;
        std::cout << "info string    fen " << row.fen << std::endl;
    }

    auto pct = [](int c, int den) -> double {
        return den > 0 ? (100.0 * double(c) / double(den)) : 0.0;
    };

    std::cout << "info string --- SUMMARY ---" << std::endl;
    std::cout << "info string valid=" << valid
              << " skipped=" << skipped
              << std::endl;

    if (valid > 0) {
        std::cout << "info string top1=" << top1
                  << " (" << pct(top1, valid) << "%)" << std::endl;
        std::cout << "info string top3=" << top3
                  << " (" << pct(top3, valid) << "%)" << std::endl;
        std::cout << "info string top5=" << top5
                  << " (" << pct(top5, valid) << "%)" << std::endl;
        std::cout << "info string mean_rank="
                  << (double(sum_rank) / double(valid)) << std::endl;
        std::cout << "info string mrr="
                  << (sum_recip_rank / double(valid)) << std::endl;

        std::cout << "info string rank_hist:";
        for (int r = 1; r <= 20; ++r) {
            if (hist[r]) std::cout << " " << r << ":" << hist[r];
        }
        if (hist[21]) std::cout << " 21+:" << hist[21];
        std::cout << std::endl;

        if (open_n > 0) {
            std::cout << "info string openings(n=" << open_n << ")"
                      << " top1=" << pct(open_top1, open_n) << "%"
                      << " top3=" << pct(open_top3, open_n) << "%"
                      << std::endl;
        }
        if (mid_n > 0) {
            std::cout << "info string middlegames(n=" << mid_n << ")"
                      << " top1=" << pct(mid_top1, mid_n) << "%"
                      << " top3=" << pct(mid_top3, mid_n) << "%"
                      << std::endl;
        }
    }

    std::cout << "info string === END POLICY HIT BENCH ===" << std::endl;
    std::cout.flush();
}