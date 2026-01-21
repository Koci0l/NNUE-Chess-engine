#pragma once

#include "config.h"
#include <cstring>
#include <vector>

namespace chess {
    class Board;
    class Piece;
    class Square;
    class Move;
}

#if defined(__x86_64__) || defined(__amd64__)
    #define CACHE_LINE_ALIGNMENT alignas(64)
#else
    #define CACHE_LINE_ALIGNMENT alignas(16)
#endif

class CACHE_LINE_ALIGNMENT Accumulator {
public:
    i16 values[HL_SIZE]{};
    i16& operator[](usize index) { return values[index]; }
    const i16& operator[](usize index) const { return values[index]; }
};

struct AccumulatorPair {
    Accumulator white;
    Accumulator black;

    void resetAccumulators(const chess::Board& board);
    void add_piece(const chess::Piece& p, const chess::Square& sq);
    void remove_piece(const chess::Piece& p, const chess::Square& sq);
    void move_piece(const chess::Piece& p, const chess::Square& from, const chess::Square& to);

    bool operator==(const AccumulatorPair& other) const {
        return std::memcmp(this, &other, sizeof(AccumulatorPair)) == 0;
    }
    bool operator!=(const AccumulatorPair& other) const {
        return !(*this == other);
    }
};

// Stack-based accumulator management
class AccumulatorStack {
private:
    static constexpr size_t MAX_DEPTH = 128;
    alignas(64) AccumulatorPair stack[MAX_DEPTH];
    size_t idx = 0;

public:
    AccumulatorPair& current() { return stack[idx]; }
    const AccumulatorPair& current() const { return stack[idx]; }

    void push() { 
        stack[idx + 1] = stack[idx];  // Same behavior as before
        ++idx;
    }

    void pop() { 
        if (idx > 0) --idx; 
    }

    void resetAccumulators(const chess::Board& board) {
        idx = 0;
        stack[0].resetAccumulators(board);
    }

    size_t size() const { return idx + 1; }
};