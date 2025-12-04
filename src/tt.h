#pragma once

#include "chess.hpp"
#include <cstdint>

enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

struct TTEntry {
    uint64_t key = 0;
    int16_t depth = -1;
    int16_t score = 0;
    chess::Move best_move;
    TTFlag flag = TT_EXACT;
    bool pv = false;
};

void initTT(size_t mb);
void clearTT();
void storeTT(uint64_t key, int depth, int score, chess::Move best_move, 
             TTFlag flag, int ply_from_root, bool pv = false);
bool probeTT(uint64_t key, int depth, int alpha, int beta, int& score, 
             chess::Move& tt_move, int ply_from_root, bool& tt_pv);
bool peekTT(uint64_t key, TTEntry& out);

extern TTEntry* tt;
extern size_t TT_SIZE;
extern size_t TT_MASK;