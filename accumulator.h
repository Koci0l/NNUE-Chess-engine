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
    std::vector<AccumulatorPair> stack;
    
public:
    AccumulatorStack() {
        stack.reserve(128); // Reserve space for deep searches
    }
    
    AccumulatorPair& current() { 
        return stack.back(); 
    }
    
    const AccumulatorPair& current() const { 
        return stack.back(); 
    }
    
    // Just copy the current accumulator (incremental updates will be applied separately)
    void push() { 
        stack.push_back(stack.back()); // Copy current state
    }
    
    void pop() { 
        if (stack.size() > 1) {
            stack.pop_back(); 
        }
    }
    
    void resetAccumulators(const chess::Board& board) {
        stack.clear();
        stack.emplace_back();
        stack.back().resetAccumulators(board);
    }
    
    size_t size() const { return stack.size(); }
};