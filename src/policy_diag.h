#pragma once

#include "chess.hpp"
#include "types.h"   // ThreadInfo

#include <cstdint>

// Pure policy ranking of all legal moves (by logit, high = better).
// out_order[i] = index into legal, best-first.
// out_logits must hold at least legal.size() floats.
// Returns false if policy not loaded / empty / forward failed.
bool policyRankLegalMoves(const chess::Board& board,
                          const chess::Movelist& legal,
                          int* out_order,
                          float* out_logits,
                          int& n);

// Policy top-N for current board (same as g_policy.debugPosition).
void runPolicyDebug(const chess::Board& board, int topN = 16);

// Hit-rate bench: pure policy rank vs search best move on a fixed FEN set.
// search_depth > 0  → fixed depth (preferred)
// search_nodes > 0 and search_depth <= 0 → node limit
// Default if both zero: depth 10
void runPolicyHitBench(int search_depth, int64_t search_nodes,
                       ThreadInfo& thread);