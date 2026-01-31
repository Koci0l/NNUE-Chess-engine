#pragma once

#include "config.h"
#include <cstring>
#include <array>

namespace chess {
    class Board;
    class Piece;
    class Square;
    class Move;
}

#if defined(__AVX512F__)
    #define ALIGNMENT alignas(64)
#elif defined(__AVX2__) || defined(__AVX__)
    #define ALIGNMENT alignas(32)
#else
    #define ALIGNMENT alignas(16)
#endif

class ALIGNMENT Accumulator {
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

class AccumulatorStack {
private:
    static constexpr size_t MAX_DEPTH = 128;
    ALIGNMENT AccumulatorPair stack[MAX_DEPTH];
    size_t idx = 0;

public:
    AccumulatorPair& current() { return stack[idx]; }
    const AccumulatorPair& current() const { return stack[idx]; }

    void push() { 
        stack[idx + 1] = stack[idx];
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