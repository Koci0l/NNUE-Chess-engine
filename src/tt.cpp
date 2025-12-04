#include "tt.h"
#include "types.h"
#include <cstring>
#include <iostream>

TTEntry* tt = nullptr;
size_t TT_SIZE = 1 << 20;
size_t TT_MASK = TT_SIZE - 1;

void initTT(size_t mb) {
    size_t bytes = mb * 1024 * 1024;
    size_t entries = bytes / sizeof(TTEntry);
    size_t power = 1;
    while (power * 2 <= entries) power *= 2;
    TT_SIZE = power;
    TT_MASK = TT_SIZE - 1;
    delete[] tt;
    tt = new TTEntry[TT_SIZE];
    std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    std::cout << "info string Hash table initialized: " << mb << " MB (" 
              << TT_SIZE << " entries)" << std::endl;
}

void clearTT() {
    if (tt) std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
}

void storeTT(uint64_t key, int depth, int score, chess::Move best_move, 
             TTFlag flag, int ply_from_root, bool pv) {
    size_t index = key & TT_MASK;
    TTEntry& entry = tt[index];

    int stored_score = score;
    if (score >= MATE_SCORE - 100) stored_score = score + ply_from_root;
    else if (score <= -MATE_SCORE + 100) stored_score = score - ply_from_root;

    if (entry.key == 0 || entry.key == key || depth >= entry.depth + 3) {
        entry.key = key;
        entry.depth = depth;
        entry.score = stored_score;
        entry.best_move = best_move;
        entry.flag = flag;
        entry.pv = pv;
    }
}

bool probeTT(uint64_t key, int depth, int alpha, int beta, int& score, 
             chess::Move& tt_move, int ply_from_root, bool& tt_pv) {
    size_t index = key & TT_MASK;
    const TTEntry& entry = tt[index];

    if (entry.key != key) {
        tt_move = chess::Move();
        tt_pv = false;
        return false;
    }

    tt_move = entry.best_move;
    tt_pv = entry.pv;

    if (entry.depth >= depth) {
        int retrieved_score = entry.score;
        if (retrieved_score >= MATE_SCORE - 100) retrieved_score -= ply_from_root;
        else if (retrieved_score <= -MATE_SCORE + 100) retrieved_score += ply_from_root;

        if (entry.flag == TT_EXACT) { score = retrieved_score; return true; }
        if (entry.flag == TT_LOWER && retrieved_score >= beta) { score = beta; return true; }
        if (entry.flag == TT_UPPER && retrieved_score <= alpha) { score = alpha; return true; }
    }
    return false;
}

bool peekTT(uint64_t key, TTEntry& out) {
    const TTEntry& e = tt[key & TT_MASK];
    if (e.key != key) return false;
    out = e;
    return true;
}