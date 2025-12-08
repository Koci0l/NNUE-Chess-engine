#include "zobrist.h"

// Optimization: Use the chess library's built-in incremental Zobrist hashing.
// Previous implementation recalculated the hash from scratch (O(64) complexity).
// board.zobrist() returns the pre-calculated hash instantly (O(1) complexity).

void initZobrist() {
    // No initialization needed anymore.
    // The library handles its own keys internally.
    // Kept empty to maintain compatibility with uci.cpp calling convention.
}

uint64_t getZobristHash(const chess::Board& board) {
    return board.zobrist();
}