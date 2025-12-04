#include "timeman.h"
#include <algorithm>

void TimeManager::init(int time_ms, int inc_ms, int movestogo, int fixed_movetime) {
    time_left_ms = time_ms;
    increment_ms = inc_ms;
    movetime_ms = fixed_movetime;
    stability_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    
    if (movetime_ms > 0) {
        soft_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;
        hard_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;
    } else if (time_left_ms > 0) {
        int base_time = (time_left_ms / 20) + (increment_ms / 2);
        if (movestogo > 0 && movestogo <= 10)
            base_time = std::min(base_time, time_left_ms / (movestogo + 2));
        soft_limit_ms = base_time;
        hard_limit_ms = std::min(base_time * 4, time_left_ms / 3);
        soft_limit_ms = std::max(MIN_THINKING_TIME, 
                                  std::min(soft_limit_ms, time_left_ms - MOVE_OVERHEAD_MS));
        hard_limit_ms = std::max(soft_limit_ms, 
                                  std::min(hard_limit_ms, time_left_ms - MOVE_OVERHEAD_MS));
    } else {
        soft_limit_ms = 999999;
        hard_limit_ms = 999999;
    }
}

int64_t TimeManager::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
}

bool TimeManager::should_stop() const { 
    return elapsed_ms() >= hard_limit_ms; 
}

bool TimeManager::should_continue_depth(int /*depth*/, double last_depth_ms) const {
    int64_t elapsed = elapsed_ms();
    if (elapsed >= soft_limit_ms) return false;
    double estimated_next = last_depth_ms * 3.5;
    if (elapsed + estimated_next > hard_limit_ms) return false;
    return true;
}

void TimeManager::update_stability(chess::Move best_move) {
    if (best_move == last_best_move) stability_count++;
    else stability_count = 0;
    last_best_move = best_move;
}