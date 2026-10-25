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

// The feature-transformed weights are read with *aligned* SIMD loads
// (_mm512_load_si512 / _mm256_load_si256 / _mm_load_si128), so the
// accumulator must be aligned to the widest vector we might use.
//
// NOTE: the previous version tested for `AVX512F` / `AVX2` / `AVX` which are
// never defined by the compiler (the real macros are __AVX512F__ etc.), so the
// alignment silently fell back to 16 and every AVX2/AVX512 build performed
// misaligned aligned-loads -> SIGSEGV (#GP) or garbage.  Just use 64 bytes
// unconditionally: Accumulator is 2048 bytes so this costs nothing.
#define ALIGNMENT alignas(64)

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
    void add_piece(const chess::Piece& p, const chess::Square& sq, chess::Square wk, chess::Square bk);
    void remove_piece(const chess::Piece& p, const chess::Square& sq, chess::Square wk, chess::Square bk);
    void move_piece(const chess::Piece& p, const chess::Square& from, const chess::Square& to, chess::Square wk, chess::Square bk);

    bool operator==(const AccumulatorPair& other) const {
        return std::memcmp(this, &other, sizeof(AccumulatorPair)) == 0;
    }
    bool operator!=(const AccumulatorPair& other) const {
        return !(*this == other);
    }
};

class AccumulatorStack {
private:
    // One accumulator is needed per ply, plus one more for the singular-extension
    // verification search which is performed at the same ply as the node itself.
    // MAX_PLY is 100, so 256 entries (1 MiB) is always enough; the previous value
    // of 128 could be exceeded (-> out-of-bounds writes into the object) on long
    // forcing lines with singular extensions.
    static constexpr size_t MAX_DEPTH = 256;
    ALIGNMENT AccumulatorPair stack[MAX_DEPTH];
    size_t idx = 0;

public:
    AccumulatorPair& current() { return stack[idx]; }
    const AccumulatorPair& current() const { return stack[idx]; }

    void push() {
        if (idx + 1 >= MAX_DEPTH) return; // never happens, but never corrupt memory either
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