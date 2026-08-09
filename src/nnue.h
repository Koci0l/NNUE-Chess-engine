#pragma once
#include "config.h"
#include "accumulator.h"
#include "chess.hpp"
#include <array>
#include <string>
// Include types.h for ThreadInfo
#include "types.h"

#if defined(AVX512F)
constexpr usize NN_ALIGNMENT = 64;
#elif defined(AVX2) || defined(AVX)
constexpr usize NN_ALIGNMENT = 32;
#else
constexpr usize NN_ALIGNMENT = 16;
#endif

struct InputBucketInfo {
    usize bucket;
    int flip;
};

struct NNUE {
    // Weights and biases (expanded weightsToHL for input buckets)
    alignas(NN_ALIGNMENT) std::array<i16, INPUT_SIZE * HL_SIZE * NUM_INPUT_BUCKETS> weightsToHL;
    alignas(NN_ALIGNMENT) std::array<i16, HL_SIZE> hiddenLayerBias;
    alignas(NN_ALIGNMENT) std::array<std::array<i16, 2 * HL_SIZE>, OUTPUT_BUCKETS> weightsToOut;
    std::array<i16, OUTPUT_BUCKETS> outputBias;

    static i16 ReLU(const i16 x);
    static i16 CReLU(const i16 x);
    static i32 SCReLU(const i16 x);
<<<<<<< Updated upstream
    
    i32 vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket);
    
    static InputBucketInfo getInputBucketInfo(chess::Square ksq);
    static usize feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square, usize bucket, int flip);
    
    static usize getMaterialBucket(const chess::Board& board);
    
=======
    i32 vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket);
    static usize feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square);
    static int get_input_bucket(chess::Color perspective, chess::Square king_sq); // Added
    static usize getMaterialBucket(const chess::Board& board);
>>>>>>> Stashed changes
    void loadNetwork(const std::string& filepath);
    int forwardPass(const chess::Board* board, const AccumulatorPair& accumulators);
    static i16 evaluate(const chess::Board& board, ThreadInfo& thisThread);
    void debugNetwork(const chess::Board& board, const AccumulatorPair& accumulators);
    void debugVectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket);
    void showBuckets(const chess::Board* board, const AccumulatorPair& accumulators);
};

extern NNUE g_nnue;