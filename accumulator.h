#pragma once

#include "config.h"
#include <cstring>

namespace chess {
    class Board;
    class Piece;
    class Square;
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

    bool operator==(const AccumulatorPair& other) const {
        return std::memcmp(this, &other, sizeof(AccumulatorPair)) == 0;
    }
    bool operator!=(const AccumulatorPair& other) const {
        return !(*this == other);
    }
};