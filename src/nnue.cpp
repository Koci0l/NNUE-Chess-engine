#include "nnue.h"
#include "config.h"
#include "accumulator.h"
#include "chess.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

NNUE g_nnue;

// Bucket layout for ChessBucketsMirrored (Files A-D mirrored to H-E)
constexpr int BUCKET_LAYOUT[32] = {
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
};

// ============================================================================
// Bucket & Feature Calculation
// ============================================================================
int NNUE::get_input_bucket(chess::Color perspective, chess::Square king_sq) {
    int sq = king_sq.index();
    if (perspective == chess::Color::BLACK) {
        sq ^= 56; // flip rank for black's perspective
    }
    int rank = sq / 8;
    int file = sq % 8;
    if (file >= 4) {
        file = 7 - file; // mirror file (left/right symmetry)
    }
    return BUCKET_LAYOUT[rank * 4 + file];
}

usize NNUE::feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square) {
    const int colorIndex = (perspective == color) ? 0 : 1;
    const int squareIndex = (perspective == chess::Color::BLACK) ? (square ^ 56).index() : square.index();
    return colorIndex * 384 + static_cast<int>(piece) * 64 + squareIndex;
}

usize NNUE::getMaterialBucket(const chess::Board& board) {
    constexpr usize divisor = 32 / OUTPUT_BUCKETS;
    const int pieceCount = board.occ().count();
    return static_cast<usize>((pieceCount - 2) / divisor);
}

// ============================================================================
// Accumulator Updates (with Bucket Indexing)
// ============================================================================
void AccumulatorPair::add_piece(const chess::Piece& p, const chess::Square& sq, bool skip_white, bool skip_black) {
    if (p == chess::Piece::NONE) return;
    const usize featureW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
    const usize featureB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
    const usize offsetW = white_bucket * INPUT_SIZE * HL_SIZE + featureW * HL_SIZE;
    const usize offsetB = black_bucket * INPUT_SIZE * HL_SIZE + featureB * HL_SIZE;
    for (usize i = 0; i < HL_SIZE; ++i) {
        if (!skip_white) white.values[i] += g_nnue.weightsToHL[offsetW + i];
        if (!skip_black) black.values[i] += g_nnue.weightsToHL[offsetB + i];
    }
}

void AccumulatorPair::remove_piece(const chess::Piece& p, const chess::Square& sq, bool skip_white, bool skip_black) {
    if (p == chess::Piece::NONE) return;
    const usize featureW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
    const usize featureB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
    const usize offsetW = white_bucket * INPUT_SIZE * HL_SIZE + featureW * HL_SIZE;
    const usize offsetB = black_bucket * INPUT_SIZE * HL_SIZE + featureB * HL_SIZE;
    for (usize i = 0; i < HL_SIZE; ++i) {
        if (!skip_white) white.values[i] -= g_nnue.weightsToHL[offsetW + i];
        if (!skip_black) black.values[i] -= g_nnue.weightsToHL[offsetB + i];
    }
}

void AccumulatorPair::move_piece(const chess::Piece& p, const chess::Square& from, const chess::Square& to, bool skip_white, bool skip_black) {
    if (p == chess::Piece::NONE) return;
    const usize featureFromW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), from);
    const usize featureFromB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), from);
    const usize featureToW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), to);
    const usize featureToB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), to);
    
    const usize offsetFromW = white_bucket * INPUT_SIZE * HL_SIZE + featureFromW * HL_SIZE;
    const usize offsetToW = white_bucket * INPUT_SIZE * HL_SIZE + featureToW * HL_SIZE;
    const usize offsetFromB = black_bucket * INPUT_SIZE * HL_SIZE + featureFromB * HL_SIZE;
    const usize offsetToB = black_bucket * INPUT_SIZE * HL_SIZE + featureToB * HL_SIZE;

    for (usize i = 0; i < HL_SIZE; ++i) {
        if (!skip_white) white.values[i] += g_nnue.weightsToHL[offsetToW + i] - g_nnue.weightsToHL[offsetFromW + i];
        if (!skip_black) black.values[i] += g_nnue.weightsToHL[offsetToB + i] - g_nnue.weightsToHL[offsetFromB + i];
    }
}

void AccumulatorPair::refresh_white(const chess::Board& board) {
    std::memcpy(white.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));
    chess::Square wk_sq = board.kingSq(chess::Color::WHITE);
    white_bucket = NNUE::get_input_bucket(chess::Color::WHITE, wk_sq);
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece p = board.at(sq);
        if (p != chess::Piece::NONE) {
            const usize featureW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
            const usize offsetW = white_bucket * INPUT_SIZE * HL_SIZE + featureW * HL_SIZE;
            for (usize i = 0; i < HL_SIZE; ++i) {
                white.values[i] += g_nnue.weightsToHL[offsetW + i];
            }
        }
    }
}

void AccumulatorPair::refresh_black(const chess::Board& board) {
    std::memcpy(black.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));
    chess::Square bk_sq = board.kingSq(chess::Color::BLACK);
    black_bucket = NNUE::get_input_bucket(chess::Color::BLACK, bk_sq);
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece p = board.at(sq);
        if (p != chess::Piece::NONE) {
            const usize featureB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
            const usize offsetB = black_bucket * INPUT_SIZE * HL_SIZE + featureB * HL_SIZE;
            for (usize i = 0; i < HL_SIZE; ++i) {
                black.values[i] += g_nnue.weightsToHL[offsetB + i];
            }
        }
    }
}

void AccumulatorPair::resetAccumulators(const chess::Board& board) {
    std::memcpy(white.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));
    std::memcpy(black.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));
    
    chess::Square wk_sq = board.kingSq(chess::Color::WHITE);
    chess::Square bk_sq = board.kingSq(chess::Color::BLACK);
    white_bucket = NNUE::get_input_bucket(chess::Color::WHITE, wk_sq);
    black_bucket = NNUE::get_input_bucket(chess::Color::BLACK, bk_sq);
    
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece p = board.at(sq);
        if (p != chess::Piece::NONE) {
            add_piece(p, sq);
        }
    }
}

// ============================================================================
// Activation Functions
// ============================================================================
i16 NNUE::ReLU(const i16 x) {
    return (x < 0) ? 0 : x;
}

i16 NNUE::CReLU(const i16 x) {
    if (x < 0) return 0;
    if (x > QA) return QA;
    return x;
}

i32 NNUE::SCReLU(const i16 x) {
    const i32 clamped = std::clamp(static_cast<i32>(x), 0, static_cast<i32>(QA));
    return clamped * clamped;
}

// ============================================================================
// Vectorized SCReLU Forward Pass
// ============================================================================
#if defined(__x86_64__) || defined(__amd64__) || (defined(_WIN64) && (defined(_M_X64) || defined(_M_AMD64)))
#include <immintrin.h>

#if defined(__AVX512F__)
#pragma message("Using AVX512 NNUE inference")
using Vectori16 = __m512i;
using Vectori32 = __m512i;
#define vec_set1_epi16 _mm512_set1_epi16
#define vec_load_epi16(x) _mm512_load_si512(reinterpret_cast<const Vectori16*>(x))
#define vec_min_epi16 _mm512_min_epi16
#define vec_max_epi16 _mm512_max_epi16
#define vec_madd_epi16 _mm512_madd_epi16
#define vec_mullo_epi16 _mm512_mullo_epi16
#define vec_add_epi32 _mm512_add_epi32
#define vec_setzero_epi32() _mm512_setzero_si512()
inline i32 vec_reduce_epi32(Vectori32 vec) {
    return _mm512_reduce_add_epi32(vec);
}
#elif defined(__AVX2__)
#pragma message("Using AVX2 NNUE inference")
using Vectori16 = __m256i;
using Vectori32 = __m256i;
#define vec_set1_epi16 _mm256_set1_epi16
#define vec_load_epi16(x) _mm256_load_si256(reinterpret_cast<const Vectori16*>(x))
#define vec_min_epi16 _mm256_min_epi16
#define vec_max_epi16 _mm256_max_epi16
#define vec_madd_epi16 _mm256_madd_epi16
#define vec_mullo_epi16 _mm256_mullo_epi16
#define vec_add_epi32 _mm256_add_epi32
#define vec_setzero_epi32() _mm256_setzero_si256()
inline i32 vec_reduce_epi32(Vectori32 vec) {
    __m128i xmm1 = _mm256_extracti128_si256(vec, 1);
    __m128i xmm0 = _mm256_castsi256_si128(vec);
    xmm0 = _mm_add_epi32(xmm0, xmm1);
    xmm1 = _mm_shuffle_epi32(xmm0, 0xEE);
    xmm0 = _mm_add_epi32(xmm0, xmm1);
    xmm1 = _mm_shuffle_epi32(xmm0, 0x55);
    xmm0 = _mm_add_epi32(xmm0, xmm1);
    return _mm_cvtsi128_si32(xmm0);
}
#else
#pragma message("Using SSE NNUE inference")
using Vectori16 = __m128i;
using Vectori32 = __m128i;
#define vec_set1_epi16 _mm_set1_epi16
#define vec_load_epi16(x) _mm_load_si128(reinterpret_cast<const Vectori16*>(x))
#define vec_min_epi16 _mm_min_epi16
#define vec_max_epi16 _mm_max_epi16
#define vec_madd_epi16 _mm_madd_epi16
#define vec_mullo_epi16 _mm_mullo_epi16
#define vec_add_epi32 _mm_add_epi32
#define vec_setzero_epi32() _mm_setzero_si128()
inline i32 vec_reduce_epi32(Vectori32 vec) {
    __m128i xmm1 = _mm_shuffle_epi32(vec, 0xEE);
    vec = _mm_add_epi32(vec, xmm1);
    xmm1 = _mm_shuffle_epi32(vec, 0x55);
    vec = _mm_add_epi32(vec, xmm1);
    return _mm_cvtsi128_si32(vec);
}
#endif

i32 NNUE::vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    constexpr usize VECTOR_SIZE = sizeof(Vectori16) / sizeof(i16);
    static_assert(HL_SIZE % VECTOR_SIZE == 0, "HL_SIZE must be divisible by vector size");

    const Vectori16 VEC_QA = vec_set1_epi16(QA);
    const Vectori16 VEC_ZERO = vec_set1_epi16(0);
    Vectori32 accumulator = vec_setzero_epi32();

    for (usize i = 0; i < HL_SIZE; i += VECTOR_SIZE) {
        const Vectori16 stmValues = vec_load_epi16(&stm.values[i]);
        const Vectori16 nstmValues = vec_load_epi16(&nstm.values[i]);

        const Vectori16 stmClamped = vec_min_epi16(VEC_QA, vec_max_epi16(stmValues, VEC_ZERO));
        const Vectori16 nstmClamped = vec_min_epi16(VEC_QA, vec_max_epi16(nstmValues, VEC_ZERO));

        const Vectori16 stmWeights = vec_load_epi16(&weightsToOut[bucket][i]);
        const Vectori16 nstmWeights = vec_load_epi16(&weightsToOut[bucket][i + HL_SIZE]);

        const Vectori32 stmActivated = vec_madd_epi16(stmClamped, vec_mullo_epi16(stmClamped, stmWeights));
        const Vectori32 nstmActivated = vec_madd_epi16(nstmClamped, vec_mullo_epi16(nstmClamped, nstmWeights));

        accumulator = vec_add_epi32(accumulator, stmActivated);
        accumulator = vec_add_epi32(accumulator, nstmActivated);
    }
    return vec_reduce_epi32(accumulator);
}

#else
#pragma message("Using scalar NNUE inference")
i32 NNUE::vectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    i32 res = 0;
    for (usize i = 0; i < HL_SIZE; i++) {
        res += SCReLU(stm.values[i]) * weightsToOut[bucket][i];
        res += SCReLU(nstm.values[i]) * weightsToOut[bucket][i + HL_SIZE];
    }
    return res;
}
#endif

// ============================================================================
// Network Loading (Updated Expected Size calculation)
// ============================================================================
void NNUE::loadNetwork(const std::string& filepath) {
    std::ifstream stream(filepath, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "ERROR: Failed to open network file: " << filepath << std::endl;
        return;
    }
    stream.seekg(0, std::ios::end);
    size_t fileSize = stream.tellg();
    stream.seekg(0, std::ios::beg);
    
    // Included NUM_INPUT_BUCKETS in expected size check
    size_t expectedSize = sizeof(i16) * (INPUT_SIZE * HL_SIZE * NUM_INPUT_BUCKETS + HL_SIZE + 2 * HL_SIZE * OUTPUT_BUCKETS + OUTPUT_BUCKETS);
    std::cout << "Network file size: " << fileSize << " bytes" << std::endl;
    std::cout << "Expected size: " << expectedSize << " bytes" << std::endl;

    stream.read(reinterpret_cast<char*>(weightsToHL.data()), weightsToHL.size() * sizeof(i16));
    stream.read(reinterpret_cast<char*>(hiddenLayerBias.data()), hiddenLayerBias.size() * sizeof(i16));
    for (usize bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {
        stream.read(reinterpret_cast<char*>(weightsToOut[bucket].data()), weightsToOut[bucket].size() * sizeof(i16));
    }
    stream.read(reinterpret_cast<char*>(outputBias.data()), outputBias.size() * sizeof(i16));

    if (!stream) std::cerr << "ERROR: Malformed or incomplete network file: " << filepath << std::endl;
    else std::cout << "NNUE file loaded successfully: " << filepath << std::endl;
}

// ============================================================================
// Forward Pass & Evaluation
// ============================================================================
int NNUE::forwardPass(const chess::Board* board, const AccumulatorPair& accumulators) {
    const usize outputBucket = getMaterialBucket(*board);
    const bool isWhiteSTM = board->sideToMove() == chess::Color::WHITE;
    const Accumulator& accumulatorSTM = isWhiteSTM ? accumulators.white : accumulators.black;
    const Accumulator& accumulatorNSTM = isWhiteSTM ? accumulators.black : accumulators.white;
    
    i64 eval = vectorizedSCReLU(accumulatorSTM, accumulatorNSTM, outputBucket);
    
    // Dequantization for SCReLU
    eval /= QA;
    eval += outputBias[outputBucket];
    return static_cast<int>((eval * EVAL_SCALE) / (static_cast<i64>(QA) * QB));
}

i16 NNUE::evaluate(const chess::Board& board, ThreadInfo& thisThread) {
    const int eval = g_nnue.forwardPass(&board, thisThread.accumulatorStack.current());
    return std::clamp(eval, Search::TB_MATED_IN_MAX_PLY, Search::TB_MATE_IN_MAX_PLY);
}

// ============================================================================
// Debug Functions (Kept identical)
// ============================================================================
void NNUE::debugVectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    i64 stm_contrib = 0, nstm_contrib = 0;
    for (usize i = 0; i < HL_SIZE; i++) {
        stm_contrib += static_cast<i64>(SCReLU(stm.values[i])) * weightsToOut[bucket][i];
        nstm_contrib += static_cast<i64>(SCReLU(nstm.values[i])) * weightsToOut[bucket][i + HL_SIZE];
    }
    std::cout << "STM contribution: " << stm_contrib << std::endl;
    std::cout << "NSTM contribution: " << nstm_contrib << std::endl;
    std::cout << "Total: " << (stm_contrib + nstm_contrib) << std::endl;
}

void NNUE::debugNetwork(const chess::Board& board, const AccumulatorPair& accumulators) {
    std::cout << "\n========== NNUE DEBUG ==========\n" << std::endl;
    const usize bucket = getMaterialBucket(board);
    const bool isWhiteSTM = board.sideToMove() == chess::Color::WHITE;
    const Accumulator& stm = isWhiteSTM ? accumulators.white : accumulators.black;
    const Accumulator& nstm = isWhiteSTM ? accumulators.black : accumulators.white;
    
    i32 rawEval = vectorizedSCReLU(stm, nstm, bucket);
    std::cout << "Raw vectorizedSCReLU result: " << rawEval << std::endl;
    std::cout << "\n================================\n" << std::endl;
}

void NNUE::showBuckets(const chess::Board* board, const AccumulatorPair& accumulators) {
    std::cout << "+------------+------------+\n"
              << "|   Bucket   | Evaluation |\n"
              << "+------------+------------+" << std::endl;
    const usize currentBucket = getMaterialBucket(*board);
    const bool isWhiteSTM = board->sideToMove() == chess::Color::WHITE;
    const Accumulator& accumulatorSTM = isWhiteSTM ? accumulators.white : accumulators.black;
    const Accumulator& accumulatorNSTM = isWhiteSTM ? accumulators.black : accumulators.white;

    for (usize bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {
        i32 eval = vectorizedSCReLU(accumulatorSTM, accumulatorNSTM, bucket);
        i64 evalScaled = eval / QA;
        evalScaled += outputBias[bucket];
        int finalEval = static_cast<int>((evalScaled * EVAL_SCALE) / (static_cast<i64>(QA) * QB));
        const char* marker = (bucket == currentBucket) ? "*" : " ";
        printf("| %s%-9zu | %+10.2f |\n", marker, bucket, finalEval / 100.0);
    }
    std::cout << "+------------+------------+" << std::endl;
    std::cout << "* = active bucket (material: " << board->occ().count() << " pieces)" << std::endl;
}