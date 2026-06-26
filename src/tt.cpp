#include "tt.h"
#include "types.h"
#include <cstring>
#include <iostream>

TTEntry* tt = nullptr;
size_t TT_SIZE = 1 << 20;
size_t TT_MASK = TT_SIZE - 1;
uint8_t current_generation = 0; // NEW

void initTT(size_t mb) {
    size_t bytes = mb * 1024ULL * 1024ULL;
    size_t entries = bytes / sizeof(TTEntry);
    size_t power = 1;
    while (power * 2 <= entries) {
        power *= 2;
    }
    if (power == 0) {
        power = 1;
    }
    TT_SIZE = power;
    TT_MASK = TT_SIZE - 1;
    delete[] tt;
    tt = new TTEntry[TT_SIZE];
    std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    std::cout << "info string Hash table initialized: " << mb << " MB ("
              << TT_SIZE << " entries)" << std::endl;
}

void clearTT() {
    if (tt) {
        std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    }
}

void advanceTTGeneration() {
    current_generation++;
}

void storeTT(uint64_t key, int depth, int score, chess::Move best_move,
             TTFlag flag, int ply_from_root, bool pv) {
    if (!tt) return;
    size_t index = key & TT_MASK;
    TTEntry& entry = tt[index];

    int stored_score = score;
    if (score >= MATE_SCORE - 100) {
        stored_score = score + ply_from_root;
    } else if (score <= -MATE_SCORE + 100) {
        stored_score = score - ply_from_root;
    }

    uint16_t move_to_store = best_move.move();
    if (move_to_store == 0 && entry.key == key) {
        move_to_store = entry.best_move;
    }

    bool replace = false;
    if (entry.key == 0) {
        replace = true; // Empty slot
    } else if (entry.key == key) {
        // Same position: Always update if depth is close or if it's a PV node
        replace = (depth >= entry.depth - 4 || pv);
    } else {
        // Collision: Use a priority score. 
        int age_diff = (uint8_t)(current_generation - entry.generation);
        int entry_priority = entry.depth - (age_diff * 2); 
        replace = (depth >= entry_priority);
    }

    if (replace) {
        entry.key = key;
        entry.depth = static_cast<int8_t>(depth); // FIX: was int16_t cast
        entry.score = static_cast<int16_t>(stored_score);
        entry.best_move = move_to_store;
        entry.flag = flag;
        entry.pv = pv;
        entry.generation = current_generation;
    }
}

bool probeTT(uint64_t key, int depth, int alpha, int beta, int& score,
             chess::Move& tt_move, int ply_from_root, bool& tt_pv) {
    if (!tt) {
        tt_move = chess::Move();
        tt_pv = false;
        return false;
    }
    size_t index = key & TT_MASK;
    const TTEntry& entry = tt[index];

    if (entry.key != key) {
        tt_move = chess::Move();
        tt_pv = false;
        return false;
    }

    tt_move = chess::Move(entry.best_move); // Convert back to chess::Move
    tt_pv = entry.pv;

    if (entry.depth >= depth) {
        int retrieved_score = entry.score;
        if (retrieved_score >= MATE_SCORE - 100) {
            retrieved_score -= ply_from_root;
        } else if (retrieved_score <= -MATE_SCORE + 100) {
            retrieved_score += ply_from_root;
        }

        if (entry.flag == TT_EXACT) {
            score = retrieved_score;
            return true;
        }
        if (entry.flag == TT_LOWER && retrieved_score >= beta) {
            score = beta;
            return true;
        }
        if (entry.flag == TT_UPPER && retrieved_score <= alpha) {
            score = alpha;
            return true;
        }
    }
    return false;
}

bool peekTT(uint64_t key, TTEntry& out) {
    if (!tt) return false;
    const TTEntry& e = tt[key & TT_MASK];
    if (e.key != key) return false;
    out = e;
    return true;
}