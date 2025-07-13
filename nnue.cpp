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

#if defined(__x86_64__) || defined(__amd64__)
#include <immintrin.h>
#if defined(__AVX512F__) && defined(__AVX512BW__)
#pragma message("Using AVX512 NNUE inference")
using Vectori16 = __m512i; using Vectori32 = __m512i;
#define set1_epi16 _mm512_set1_epi16
#define load_epi16(x) _mm512_load_si512(reinterpret_cast<const Vectori16*>(x))
#define min_epi16 _mm512_min_epi16
#define max_epi16 _mm512_max_epi16
#define madd_epi16 _mm512_madd_epi16
#define mullo_epi16(a, b) _mm512_mullo_epi16(a, b)
#define add_epi32 _mm512_add_epi32
#define reduce_epi32 _mm512_reduce_add_epi32
#elif defined(__AVX2__)
#pragma message("Using AVX2 NNUE inference")
using Vectori16 = __m256i; using Vectori32 = __m256i;
#define set1_epi16 _mm256_set1_epi16
#define load_epi16(x) _mm256_load_si256(reinterpret_cast<const Vectori16*>(x))
#define min_epi16 _mm256_min_epi16
#define max_epi16 _mm256_max_epi16
#define madd_epi16 _mm256_madd_epi16
#define mullo_epi16(a, b) _mm256_mullo_epi16(a, b)
#define add_epi32 _mm256_add_epi32
#define reduce_epi32 \
    [](Vectori32 vec) { \
        const __m128i xmm1 = _mm256_extracti128_si256(vec, 1); \
        __m128i xmm0 = _mm256_castsi256_si128(vec); \
        xmm0         = _mm_add_epi32(xmm0, xmm1); \
        const __m128i xmm2 = _mm_shuffle_epi32(xmm0, 0b10110001); \
        xmm0         = _mm_add_epi32(xmm0, xmm2); \
        const __m128i xmm3 = _mm_shuffle_epi32(xmm0, 0b01010101); \
        xmm0         = _mm_add_epi32(xmm0, xmm3); \
        return _mm_cvtsi128_si32(xmm0); \
    }
#else
#pragma message("Using SSE NNUE inference")
using Vectori16 = __m128i; using Vectori32 = __m128i;
#define set1_epi16 _mm_set1_epi16
#define load_epi16(x) _mm_load_si128(reinterpret_cast<const Vectori16*>(x))
#define min_epi16 _mm_min_epi16
#define max_epi16 _mm_max_epi16
#define madd_epi16 _mm_madd_epi16
#define mullo_epi16(a, b) _mm_mullo_epi16(a, b)
#define add_epi32 _mm_add_epi32
#define reduce_epi32 \
    [](Vectori32 vec) { \
        vec = _mm_add_epi32(vec, _mm_shuffle_epi32(vec, 0b10110001)); \
        vec = _mm_add_epi32(vec, _mm_shuffle_epi32(vec, 0b01010101)); \
        return _mm_cvtsi128_si32(vec); \
    }
#endif
i32 NNUE::vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    const usize VECTOR_SIZE = sizeof(Vectori16) / sizeof(i16);
    static_assert(HL_SIZE % VECTOR_SIZE == 0, "HL_SIZE must be divisible by vector size.");
    const Vectori16 VEC_QA   = set1_epi16(QA);
    const Vectori16 VEC_ZERO = set1_epi16(0);
    Vectori32 total_sum{};
    for (usize i = 0; i < HL_SIZE; i += VECTOR_SIZE) {
        Vectori16 stm_vals  = load_epi16(&stm.values[i]);
        Vectori16 nstm_vals = load_epi16(&nstm.values[i]);
        stm_vals = min_epi16(VEC_QA, max_epi16(stm_vals, VEC_ZERO));
        nstm_vals = min_epi16(VEC_QA, max_epi16(nstm_vals, VEC_ZERO));
        const Vectori16 stm_weights = load_epi16(&weightsToOut[bucket][i]);
        const Vectori16 nstm_weights = load_epi16(&weightsToOut[bucket][i + HL_SIZE]);
        const Vectori16 stm_sq = mullo_epi16(stm_vals, stm_vals);
        const Vectori16 nstm_sq = mullo_epi16(nstm_vals, nstm_vals);
        Vectori32 stm_prod = madd_epi16(stm_sq, stm_weights);
        Vectori32 nstm_prod = madd_epi16(nstm_sq, nstm_weights);
        total_sum = add_epi32(total_sum, stm_prod);
        total_sum = add_epi32(total_sum, nstm_prod);
    }
    return reduce_epi32(total_sum);
}
#else
#pragma message("Using scalar NNUE inference")
i32 NNUE::vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    i32 res = 0;
    for (usize i = 0; i < HL_SIZE; i++) {
        res += (i32)SCReLU(stm.values[i]) * weightsToOut[bucket][i];
        res += (i32)SCReLU(nstm.values[i]) * weightsToOut[bucket][i + HL_SIZE];
    }
    return res;
}
#endif

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

    // Directly read the weights, assuming the file format is already optimized
    // for this engine's access pattern ([INPUT_SIZE][HL_SIZE]).
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
    const int eval = g_nnue.forwardPass(&board, thisThread.accumulatorStack);
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