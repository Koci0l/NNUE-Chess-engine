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
        // No time control (go depth / go nodes / go infinite)
        soft_limit_ms = 999999999;
        hard_limit_ms = 999999999;
        return;
    }

    int available = std::max(1, time_left_ms - MOVE_OVERHEAD_MS);

    int base_time;
    if (mtg > 0) {
        // Moves-to-go mode: distribute evenly with a small buffer
        base_time = available / (mtg + 1) + (inc_ms * 3 / 4);
    } else {
        // Sudden death / Fischer: estimate moves remaining from game phase
        // Early game (ply < 20) assume ~40 moves left, late game fewer
        int move_number = ply / 2 + 1; // full moves played
        int estimated_remaining = std::max(10, 50 - move_number);
        base_time = available / estimated_remaining + (inc_ms * 3 / 4);
    }

    // Clamp base_time so we never plan to use more than a fraction of remaining time
    base_time = std::min(base_time, available / 4);

    soft_limit_ms = std::max(MIN_THINKING_TIME, base_time);
    hard_limit_ms = std::max(soft_limit_ms, std::min(base_time * 3, available / 2));
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
    if (node_limit > 0) {
        return !should_stop();
    }

    int64_t elapsed = elapsed_ms();

    // Apply stability factor: if best move is stable, use less time
    double adjusted_soft = soft_limit_ms * stability_factor();

    if (elapsed >= adjusted_soft) return false;

    // Estimate if we can finish the next depth in time
    double estimated_next = last_depth_ms * 3.0;
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
    // If the best move hasn't changed for many iterations, spend less time.
    // If it's unstable, allow up to full soft limit.
    //   0 changes -> factor 0.6  (save time, we're confident)
    //   1 change  -> factor 0.75
    //   2+        -> factor 1.0  (unstable, use full time)
    if (stability_count >= 4) return 0.5;
    if (stability_count >= 3) return 0.6;
    if (stability_count >= 2) return 0.75;
    if (stability_count >= 1) return 0.85;
    return 1.0; // best move just changed — use full allocation
}