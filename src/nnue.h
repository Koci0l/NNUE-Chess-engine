#pragma once
#include "config.h"
#include "accumulator.h"
#include "chess.hpp"
#include <array>
#include <string>
// Include types.h for ThreadInfo
#include "types.h"

// Same fix as in accumulator.h: the old code tested undefined macros
// (AVX512F/AVX2/AVX instead of __AVX512F__/__AVX2__/__AVX), which made the
// weight arrays only 16-byte aligned while they are read with aligned 256/512
// bit loads.  Always 64-byte align.
constexpr usize NN_ALIGNMENT = 64;

struct NNUE {
    // Weights and biases
    alignas(NN_ALIGNMENT) std::array<i16, INPUT_SIZE * HL_SIZE> weightsToHL;
    alignas(NN_ALIGNMENT) std::array<i16, HL_SIZE> hiddenLayerBias;
    alignas(NN_ALIGNMENT) std::array<std::array<i16, 2 * HL_SIZE>, OUTPUT_BUCKETS> weightsToOut;
    std::array<i16, OUTPUT_BUCKETS> outputBias;

    static i16 ReLU(const i16 x);
    static i16 CReLU(const i16 x);
    static i32 SCReLU(const i16 x);
    i32 vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket);
    
    // Updated signature to include king squares for Horizontal Mirroring
    static usize feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square, chess::Square wk, chess::Square bk);
    
    static usize getMaterialBucket(const chess::Board& board);
    void loadNetwork(const std::string& filepath);
    int forwardPass(const chess::Board* board, const AccumulatorPair& accumulators);
    static i32 evaluate(const chess::Board& board, ThreadInfo& thisThread);
    void debugNetwork(const chess::Board& board, const AccumulatorPair& accumulators);
    void debugVectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket);
    void showBuckets(const chess::Board* board, const AccumulatorPair& accumulators);
};

extern NNUE g_nnue;