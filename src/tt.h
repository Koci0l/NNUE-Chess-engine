#pragma once
#include "chess.hpp"
#include <cstddef>
#include <cstdint>

enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

struct TTEntry {
    uint64_t key;            // 8 bytes
    uint16_t best_move;      // 2 bytes (was chess::Move which is 4 bytes)
    int16_t score;           // 2 bytes
    int8_t depth;            // 1 byte
    uint8_t generation;      // 1 byte
    TTFlag flag;             // 1 byte
    bool pv;                 // 1 byte
}; // Total size: exactly 16 bytes!

void initTT(size_t mb);
void clearTT();
void advanceTTGeneration();
void storeTT(uint64_t key, int depth, int score, chess::Move best_move,
             TTFlag flag, int ply_from_root, bool pv = false);
bool probeTT(uint64_t key, int depth, int alpha, int beta, int& score,
             chess::Move& tt_move, int ply_from_root, bool& tt_pv);
bool peekTT(uint64_t key, TTEntry& out);

extern TTEntry* tt;
extern size_t TT_SIZE;
extern size_t TT_MASK;
extern uint8_t current_generation;