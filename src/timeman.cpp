#include "timeman.h"
#include <algorithm>
#include <cmath>

void TimeManager::init(int time_ms, int inc_ms, int mtg, int fixed_movetime, int ply) {
    time_left_ms = time_ms;
    increment_ms = inc_ms;
    movetime_ms = fixed_movetime;
    movestogo = mtg;
    stability_count = 0;
    node_limit = 0;
    node_counter = nullptr;
    start_time = std::chrono::high_resolution_clock::now();

    if (movetime_ms > 0) {
        soft_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;
        hard_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;
        return;
    }

    if (time_left_ms <= 0) {
        soft_limit_ms = 999999999;
        hard_limit_ms = 999999999;
        return;
    }

    int available = std::max(1, time_left_ms - MOVE_OVERHEAD_MS);
    int move_number = ply / 2 + 1;

    int base_time;
    if (mtg > 0) {
        // Moves-to-go: distribute evenly, small buffer
        base_time = available / std::max(1, mtg + 1) + inc_ms * 3 / 4;
    } else {
        // Sudden death + Fischer increment
        // OLD: max(10, 50 - move) — way too many estimated moves remaining
        // NEW: max(8, 35 - move) — more realistic, uses time more aggressively
        int est_moves_left = std::max(8, 35 - move_number);

        // KEY FIX: We will earn increment on every future move.
        // Factor that into our budget so we actually spend our time.
        int effective_time = available + est_moves_left * inc_ms * 2 / 3;

        base_time = effective_time / est_moves_left;
    }

    // Cap: never plan to use more than 40% of remaining clock (was 25%)
    base_time = std::min(base_time, available * 2 / 5);

    soft_limit_ms = std::max(MIN_THINKING_TIME, base_time);

    // Hard limit: up to 5× base (was 3×), capped at 60% of remaining (was 50%)
    hard_limit_ms = std::max(soft_limit_ms, std::min(base_time * 5, available * 3 / 5));
}

void TimeManager::set_node_limit(int64_t nodes, const uint64_t* counter) {
    node_limit = nodes;
    node_counter = counter;
}

int64_t TimeManager::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
}

bool TimeManager::should_stop() const {
    if (node_limit > 0 && node_counter && *node_counter >= static_cast<uint64_t>(node_limit)) {
        return true;
    }
    return elapsed_ms() >= hard_limit_ms;
}

bool TimeManager::should_continue_depth(int /*depth*/, double last_depth_ms) const {
    if (node_limit > 0) return !should_stop();

    int64_t elapsed = elapsed_ms();

    // Stability-adjusted soft limit
    double adjusted_soft = soft_limit_ms * stability_factor();
    if (elapsed >= adjusted_soft) return false;

    // Estimate next depth — branching factor ~2.5 (was 3.0, too pessimistic)
    double estimated_next = last_depth_ms * 2.5;
    if (elapsed + estimated_next > hard_limit_ms) return false;

    return true;
}

void TimeManager::update_stability(chess::Move best_move) {
    if (best_move == last_best_move)
        stability_count++;
    else
        stability_count = 0;
    last_best_move = best_move;
}

double TimeManager::stability_factor() const {
    // OLD: went down to 0.5 after just 4 stable iterations — way too aggressive
    // NEW: minimum 0.65, requires 6+ stable iterations, much more gradual
    if (stability_count >= 6) return 0.65;
    if (stability_count >= 5) return 0.70;
    if (stability_count >= 4) return 0.75;
    if (stability_count >= 3) return 0.82;
    if (stability_count >= 2) return 0.90;
    if (stability_count >= 1) return 0.95;
    return 1.0;
}