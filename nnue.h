#pragma once

#include "config.h"
#include "accumulator.h"
#include "chess.hpp"

#include <string>
#include <array>
#include <vector>

struct ThreadInfo {
    AccumulatorPair accumulatorStack;
};

class NNUE {
private:
    alignas(64) std::array<i16, INPUT_SIZE * HL_SIZE> weightsToHL;
    alignas(64) std::array<i16, HL_SIZE> hiddenLayerBias;
    alignas(64) std::array<std::array<i16, 2 * HL_SIZE>, OUTPUT_BUCKETS> weightsToOut;
    alignas(64) std::array<i16, OUTPUT_BUCKETS> outputBias;

    i32 SCReLU(i16 x);
    i32 vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket);

public:
    NNUE() = default;

    static usize feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square);
    void loadNetwork(const std::string& filepath);
    int forwardPass(const chess::Board* board, const AccumulatorPair& accumulators);
    i16 evaluate(const chess::Board& board, ThreadInfo& thisThread);
    void showBuckets(const chess::Board* board, const AccumulatorPair& accumulators);

    friend void AccumulatorPair::resetAccumulators(const chess::Board& board);
    friend void AccumulatorPair::add_piece(const chess::Piece& p, const chess::Square& sq);
    friend void AccumulatorPair::remove_piece(const chess::Piece& p, const chess::Square& sq);
};

extern NNUE g_nnue;