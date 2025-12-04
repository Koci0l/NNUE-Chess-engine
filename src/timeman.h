#pragma once

#include "chess.hpp"
#include <chrono>

struct TimeManager {
    int time_left_ms = 0;
    int increment_ms = 0;
    int movetime_ms = 0;
    int soft_limit_ms = 0;
    int hard_limit_ms = 0;
    std::chrono::high_resolution_clock::time_point start_time;
    chess::Move last_best_move;
    int stability_count = 0;

    static constexpr int MOVE_OVERHEAD_MS = 50;
    static constexpr int MIN_THINKING_TIME = 10;

    void init(int time_ms, int inc_ms, int movestogo, int fixed_movetime);
    int64_t elapsed_ms() const;
    bool should_stop() const;
    bool should_continue_depth(int depth, double last_depth_ms) const;
    void update_stability(chess::Move best_move);
};