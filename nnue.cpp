#include "nnue.h"
#include "config.h"
#include "accumulator.h"
#include "chess.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iostream>

NNUE g_nnue;

i32 NNUE::SCReLU(const i16 x) {
    const i32 clamped = std::clamp((i32)x, 0, (i32)QA);
    return clamped * clamped;
}

i32 NNUE::vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    i32 res = 0;
    for (usize i = 0; i < HL_SIZE; i++) {
        res += (i32)SCReLU(stm.values[i]) * weightsToOut[bucket][i];
        res += (i32)SCReLU(nstm.values[i]) * weightsToOut[bucket][i + HL_SIZE];
    }
    return res;
}

usize NNUE::feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square) {
    const int colorIndex  = (perspective == color) ? 0 : 1;
    const int squareIndex = (perspective == chess::Color::BLACK) ? (square ^ 56).index() : square.index();
    return colorIndex * 384 + static_cast<int>(piece) * 64 + squareIndex;
}

void NNUE::loadNetwork(const std::string& filepath) {
    std::ifstream stream(filepath, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "ERROR: Failed to open network file: " << filepath << std::endl;
        return;
    }

    stream.read(reinterpret_cast<char*>(weightsToHL.data()), weightsToHL.size() * sizeof(i16));
    stream.read(reinterpret_cast<char*>(hiddenLayerBias.data()), hiddenLayerBias.size() * sizeof(i16));
    stream.read(reinterpret_cast<char*>(weightsToOut[0].data()), weightsToOut[0].size() * sizeof(i16));
    stream.read(reinterpret_cast<char*>(outputBias.data()), outputBias.size() * sizeof(i16));

    if (!stream) {
        std::cerr << "ERROR: Malformed or incomplete network file: " << filepath << std::endl;
    } else {
        std::cout << "NNUE file loaded successfully: " << filepath << std::endl;
    }
}

int NNUE::forwardPass(const chess::Board* board, const AccumulatorPair& accumulators) {
    const usize outputBucket = 0;
    const bool is_white_stm = board->sideToMove() == chess::Color::WHITE;
    const Accumulator& accumulatorSTM = is_white_stm ? accumulators.white : accumulators.black;
    const Accumulator& accumulatorOPP = is_white_stm ? accumulators.black : accumulators.white;
    i64 eval = vectorizedSCReLU(accumulatorSTM, accumulatorOPP, outputBucket);

    eval /= QA;
    eval += outputBias[outputBucket];
    
    return static_cast<int>((eval * EVAL_SCALE) / (static_cast<i64>(QA) * QB));
}

i16 NNUE::evaluate(const chess::Board& board, ThreadInfo& thisThread) {
    // Use current accumulator - do NOT reset it!
    const int eval = g_nnue.forwardPass(&board, thisThread.accumulatorStack.current());
    return std::clamp(eval, Search::TB_MATED_IN_MAX_PLY, Search::TB_MATE_IN_MAX_PLY);
}

void AccumulatorPair::add_piece(const chess::Piece& p, const chess::Square& sq) {
    if (p == chess::Piece::NONE) return;
    usize feature_w = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
    usize feature_b = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
    for(usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] += g_nnue.weightsToHL[feature_w * HL_SIZE + i];
        black.values[i] += g_nnue.weightsToHL[feature_b * HL_SIZE + i];
    }
}

void AccumulatorPair::remove_piece(const chess::Piece& p, const chess::Square& sq) {
    if (p == chess::Piece::NONE) return;
    usize feature_w = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
    usize feature_b = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
    for(usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] -= g_nnue.weightsToHL[feature_w * HL_SIZE + i];
        black.values[i] -= g_nnue.weightsToHL[feature_b * HL_SIZE + i];
    }
}

void AccumulatorPair::move_piece(const chess::Piece& p, const chess::Square& from, const chess::Square& to) {
    if (p == chess::Piece::NONE) return;
    
    // Remove from old square
    usize feature_from_w = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), from);
    usize feature_from_b = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), from);
    
    // Add to new square
    usize feature_to_w = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), to);
    usize feature_to_b = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), to);
    
    for(usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] -= g_nnue.weightsToHL[feature_from_w * HL_SIZE + i];
        white.values[i] += g_nnue.weightsToHL[feature_to_w * HL_SIZE + i];
        
        black.values[i] -= g_nnue.weightsToHL[feature_from_b * HL_SIZE + i];
        black.values[i] += g_nnue.weightsToHL[feature_to_b * HL_SIZE + i];
    }
}

void AccumulatorPair::resetAccumulators(const chess::Board& board) {
    std::memcpy(&white.values, &g_nnue.hiddenLayerBias, HL_SIZE * sizeof(i16));
    std::memcpy(&black.values, &g_nnue.hiddenLayerBias, HL_SIZE * sizeof(i16));

    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece p = board.at(sq);
        if (p != chess::Piece::NONE) {
            add_piece(p, sq);
        }
    }
}

void NNUE::showBuckets(const chess::Board* board, const AccumulatorPair& accumulators) {
    std::cout << "+------------+------------+\n"
              << "|   Bucket   | Evaluation |\n"
              << "+------------+------------+" << std::endl;
    const int staticEval = forwardPass(board, accumulators);
    printf("| %-10s | %+-9.2f |\n", "0", staticEval / 100.0);
    std::cout << "+------------+------------+" << std::endl;
}