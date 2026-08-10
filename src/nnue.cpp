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
// Material Bucket Calculation
// ============================================================================

usize NNUE::getMaterialBucket(const chess::Board& board) {
    constexpr usize divisor = 32 / OUTPUT_BUCKETS;
    const int pieceCount = board.occ().count();
    return static_cast<usize>((pieceCount - 2) / divisor);
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
        // Load accumulator values
        const Vectori16 stmValues = vec_load_epi16(&stm.values[i]);
        const Vectori16 nstmValues = vec_load_epi16(&nstm.values[i]);
        
        // Clamp to [0, QA]
        const Vectori16 stmClamped = vec_min_epi16(VEC_QA, vec_max_epi16(stmValues, VEC_ZERO));
        const Vectori16 nstmClamped = vec_min_epi16(VEC_QA, vec_max_epi16(nstmValues, VEC_ZERO));
        
        // Load weights for this bucket
        const Vectori16 stmWeights = vec_load_epi16(&weightsToOut[bucket][i]);
        const Vectori16 nstmWeights = vec_load_epi16(&weightsToOut[bucket][i + HL_SIZE]);
        
        // SCReLU: clamp^2 * weight
        // madd_epi16 computes (a[0]*b[0] + a[1]*b[1]) for pairs, giving i32 results
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
// Feature Index Calculation
// ============================================================================

usize NNUE::feature(chess::Color perspective, chess::Color color, chess::PieceType piece, chess::Square square) {
    const int colorIndex = (perspective == color) ? 0 : 1;
    const int squareIndex = (perspective == chess::Color::BLACK) 
        ? (square ^ 56).index() 
        : square.index();
    
    return colorIndex * 384 + static_cast<int>(piece) * 64 + squareIndex;
}

// ============================================================================
// Network Loading
// ============================================================================

void NNUE::loadNetwork(const std::string& filepath) {
    std::ifstream stream(filepath, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "ERROR: Failed to open network file: " << filepath << std::endl;
        return;
    }

    // Get file size for diagnostics
    stream.seekg(0, std::ios::end);
    size_t fileSize = stream.tellg();
    stream.seekg(0, std::ios::beg);

    size_t expectedSize = sizeof(i16) * (INPUT_SIZE * HL_SIZE + HL_SIZE + 2 * HL_SIZE * OUTPUT_BUCKETS + OUTPUT_BUCKETS);
    
    std::cout << "Network file size: " << fileSize << " bytes" << std::endl;
    std::cout << "Expected size: " << expectedSize << " bytes" << std::endl;

    // Load weightsToHL (768 * HL_SIZE)
    stream.read(reinterpret_cast<char*>(weightsToHL.data()), weightsToHL.size() * sizeof(i16));
    std::cout << "After weightsToHL, position: " << stream.tellg() << std::endl;
    
    // Load hiddenLayerBias (HL_SIZE)
    stream.read(reinterpret_cast<char*>(hiddenLayerBias.data()), hiddenLayerBias.size() * sizeof(i16));
    std::cout << "After hiddenLayerBias, position: " << stream.tellg() << std::endl;
    
    // Load weightsToOut - bucket by bucket!
    for (usize bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {
        stream.read(reinterpret_cast<char*>(weightsToOut[bucket].data()), 
                    weightsToOut[bucket].size() * sizeof(i16));
    }
    std::cout << "After weightsToOut, position: " << stream.tellg() << std::endl;
    
    // Load outputBias (OUTPUT_BUCKETS)
    stream.read(reinterpret_cast<char*>(outputBias.data()), outputBias.size() * sizeof(i16));
    std::cout << "After outputBias, position: " << stream.tellg() << std::endl;

    if (!stream) {
        std::cerr << "ERROR: Malformed or incomplete network file: " << filepath << std::endl;
    } else {
        std::cout << "NNUE file loaded successfully: " << filepath << std::endl;
        std::cout << "  Hidden layer size: " << HL_SIZE << std::endl;
        std::cout << "  Output buckets: " << OUTPUT_BUCKETS << std::endl;
        
        std::cout << "  Output biases: ";
        for (size_t i = 0; i < OUTPUT_BUCKETS; i++) {
            std::cout << outputBias[i] << " ";
        }
        std::cout << std::endl;
    }
}

// ============================================================================
// Forward Pass
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

// ============================================================================
// Evaluate
// ============================================================================

i16 NNUE::evaluate(const chess::Board& board, ThreadInfo& thisThread) {
    const int eval = g_nnue.forwardPass(&board, thisThread.accumulatorStack.current());
    return std::clamp(eval, Search::TB_MATED_IN_MAX_PLY, Search::TB_MATE_IN_MAX_PLY);
}

// ============================================================================
// Accumulator Updates
// ============================================================================

void AccumulatorPair::add_piece(const chess::Piece& p, const chess::Square& sq) {
    if (p == chess::Piece::NONE) return;
    
    const usize featureW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
    const usize featureB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
    
    for (usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] += g_nnue.weightsToHL[featureW * HL_SIZE + i];
        black.values[i] += g_nnue.weightsToHL[featureB * HL_SIZE + i];
    }
}

void AccumulatorPair::remove_piece(const chess::Piece& p, const chess::Square& sq) {
    if (p == chess::Piece::NONE) return;
    
    const usize featureW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), sq);
    const usize featureB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), sq);
    
    for (usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] -= g_nnue.weightsToHL[featureW * HL_SIZE + i];
        black.values[i] -= g_nnue.weightsToHL[featureB * HL_SIZE + i];
    }
}

void AccumulatorPair::move_piece(const chess::Piece& p, const chess::Square& from, const chess::Square& to) {
    if (p == chess::Piece::NONE) return;
    
    const usize featureFromW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), from);
    const usize featureFromB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), from);
    const usize featureToW = NNUE::feature(chess::Color::WHITE, p.color(), p.type(), to);
    const usize featureToB = NNUE::feature(chess::Color::BLACK, p.color(), p.type(), to);
    
    for (usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] += g_nnue.weightsToHL[featureToW * HL_SIZE + i] 
                         - g_nnue.weightsToHL[featureFromW * HL_SIZE + i];
        black.values[i] += g_nnue.weightsToHL[featureToB * HL_SIZE + i] 
                         - g_nnue.weightsToHL[featureFromB * HL_SIZE + i];
    }
}

void AccumulatorPair::resetAccumulators(const chess::Board& board) {
    // Initialize with biases
    std::memcpy(white.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));
    std::memcpy(black.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));

    // Add all pieces
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece p = board.at(sq);
        if (p != chess::Piece::NONE) {
            add_piece(p, sq);
        }
    }
}

void NNUE::debugVectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    i64 stm_contrib = 0, nstm_contrib = 0;
    
    for (usize i = 0; i < HL_SIZE; i++) {
        stm_contrib += static_cast<i64>(SCReLU(stm.values[i])) * weightsToOut[bucket][i];
        nstm_contrib += static_cast<i64>(SCReLU(nstm.values[i])) * weightsToOut[bucket][i + HL_SIZE];
    }
    
    std::cout << "STM contribution: " << stm_contrib << std::endl;
    std::cout << "NSTM contribution: " << nstm_contrib << std::endl;
    std::cout << "Total: " << (stm_contrib + nstm_contrib) << std::endl;
    
    // Convert to centipawns
    i64 stm_cp = (stm_contrib * EVAL_SCALE) / (static_cast<i64>(QA) * QA * QB);
    i64 nstm_cp = (nstm_contrib * EVAL_SCALE) / (static_cast<i64>(QA) * QA * QB);
    std::cout << "STM contribution (cp): " << stm_cp << std::endl;
    std::cout << "NSTM contribution (cp): " << nstm_cp << std::endl;
    
    // Weight sums for this bucket
    i64 stm_weight_sum = 0, nstm_weight_sum = 0;
    for (usize i = 0; i < HL_SIZE; i++) {
        stm_weight_sum += weightsToOut[bucket][i];
        nstm_weight_sum += weightsToOut[bucket][i + HL_SIZE];
    }
    std::cout << "STM weights sum (bucket " << bucket << "): " << stm_weight_sum << std::endl;
    std::cout << "NSTM weights sum (bucket " << bucket << "): " << nstm_weight_sum << std::endl;
    
    // Weight ranges
    i16 stm_min = 127, stm_max = -127, nstm_min = 127, nstm_max = -127;
    for (usize i = 0; i < HL_SIZE; i++) {
        i16 sw = weightsToOut[bucket][i];
        i16 nw = weightsToOut[bucket][i + HL_SIZE];
        stm_min = std::min(stm_min, sw);
        stm_max = std::max(stm_max, sw);
        nstm_min = std::min(nstm_min, nw);
        nstm_max = std::max(nstm_max, nw);
    }
    std::cout << "STM weights range (bucket " << bucket << "): [" << stm_min << ", " << stm_max << "]" << std::endl;
    std::cout << "NSTM weights range (bucket " << bucket << "): [" << nstm_min << ", " << nstm_max << "]" << std::endl;
}

void NNUE::debugNetwork(const chess::Board& board, const AccumulatorPair& accumulators) {
    std::cout << "\n========== NNUE DEBUG ==========\n" << std::endl;
    
    std::cout << "--- Weight Statistics ---" << std::endl;
    
    // weightsToHL stats
    i16 minHL = weightsToHL[0], maxHL = weightsToHL[0];
    i64 sumHL = 0;
    for (size_t i = 0; i < weightsToHL.size(); i++) {
        minHL = std::min(minHL, weightsToHL[i]);
        maxHL = std::max(maxHL, weightsToHL[i]);
        sumHL += weightsToHL[i];
    }
    std::cout << "weightsToHL: min=" << minHL << " max=" << maxHL 
              << " avg=" << (sumHL / (i64)weightsToHL.size()) << std::endl;
    std::cout << "  First 10: ";
    for (int i = 0; i < 10; i++) std::cout << weightsToHL[i] << " ";
    std::cout << std::endl;
    
    // hiddenLayerBias stats
    i16 minBias = hiddenLayerBias[0], maxBias = hiddenLayerBias[0];
    i64 sumBias = 0;
    for (size_t i = 0; i < hiddenLayerBias.size(); i++) {
        minBias = std::min(minBias, hiddenLayerBias[i]);
        maxBias = std::max(maxBias, hiddenLayerBias[i]);
        sumBias += hiddenLayerBias[i];
    }
    std::cout << "hiddenLayerBias: min=" << minBias << " max=" << maxBias 
              << " avg=" << (sumBias / (i64)hiddenLayerBias.size()) << std::endl;
    std::cout << "  First 10: ";
    for (int i = 0; i < 10; i++) std::cout << hiddenLayerBias[i] << " ";
    std::cout << std::endl;
    
    // weightsToOut stats (all buckets)
    i16 minOut = weightsToOut[0][0], maxOut = weightsToOut[0][0];
    i64 sumOut = 0;
    size_t totalOutWeights = 0;
    for (size_t b = 0; b < OUTPUT_BUCKETS; b++) {
        for (size_t i = 0; i < 2 * HL_SIZE; i++) {
            minOut = std::min(minOut, weightsToOut[b][i]);
            maxOut = std::max(maxOut, weightsToOut[b][i]);
            sumOut += weightsToOut[b][i];
            totalOutWeights++;
        }
    }
    std::cout << "weightsToOut: min=" << minOut << " max=" << maxOut 
              << " avg=" << (sumOut / (i64)totalOutWeights) << std::endl;
    std::cout << "  First 16 (bucket 0): ";
    for (int i = 0; i < 16; i++) std::cout << weightsToOut[0][i] << " ";
    std::cout << std::endl;
    
    std::cout << "outputBias: ";
    for (size_t i = 0; i < OUTPUT_BUCKETS; i++) std::cout << outputBias[i] << " ";
    std::cout << std::endl;
    
    std::cout << "outputBias (in cp): ";
    for (size_t i = 0; i < OUTPUT_BUCKETS; i++) {
        int cp = (outputBias[i] * EVAL_SCALE) / (QA * QB);
        std::cout << cp << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\n--- Accumulator Statistics ---" << std::endl;
    
    // White accumulator
    i16 minAccW = accumulators.white.values[0], maxAccW = accumulators.white.values[0];
    i64 sumAccW = 0;
    int positiveW = 0, negativeW = 0, zeroW = 0;
    for (size_t i = 0; i < HL_SIZE; i++) {
        i16 v = accumulators.white.values[i];
        minAccW = std::min(minAccW, v);
        maxAccW = std::max(maxAccW, v);
        sumAccW += v;
        if (v > 0) positiveW++;
        else if (v < 0) negativeW++;
        else zeroW++;
    }
    std::cout << "White accumulator: min=" << minAccW << " max=" << maxAccW 
              << " avg=" << (sumAccW / (i64)HL_SIZE) << std::endl;
    std::cout << "  Positive: " << positiveW << " Negative: " << negativeW << " Zero: " << zeroW << std::endl;
    std::cout << "  First 10: ";
    for (int i = 0; i < 10; i++) std::cout << accumulators.white.values[i] << " ";
    std::cout << std::endl;
    
    // Black accumulator
    i16 minAccB = accumulators.black.values[0], maxAccB = accumulators.black.values[0];
    i64 sumAccB = 0;
    for (size_t i = 0; i < HL_SIZE; i++) {
        i16 v = accumulators.black.values[i];
        minAccB = std::min(minAccB, v);
        maxAccB = std::max(maxAccB, v);
        sumAccB += v;
    }
    std::cout << "Black accumulator: min=" << minAccB << " max=" << maxAccB 
              << " avg=" << (sumAccB / (i64)HL_SIZE) << std::endl;
    std::cout << "  First 10: ";
    for (int i = 0; i < 10; i++) std::cout << accumulators.black.values[i] << " ";
    std::cout << std::endl;
    
    bool symmetric = true;
    for (size_t i = 0; i < HL_SIZE; i++) {
        if (accumulators.white.values[i] != accumulators.black.values[i]) {
            symmetric = false;
            break;
        }
    }
    std::cout << "Accumulators symmetric: " << (symmetric ? "YES" : "NO") << std::endl;
    
    std::cout << "\n--- Forward Pass Breakdown ---" << std::endl;
    
    const usize bucket = getMaterialBucket(board);
    std::cout << "Piece count: " << board.occ().count() << std::endl;
    std::cout << "Material bucket: " << bucket << std::endl;
    
    const bool isWhiteSTM = board.sideToMove() == chess::Color::WHITE;
    std::cout << "Side to move: " << (isWhiteSTM ? "WHITE" : "BLACK") << std::endl;
    
    const Accumulator& stm = isWhiteSTM ? accumulators.white : accumulators.black;
    const Accumulator& nstm = isWhiteSTM ? accumulators.black : accumulators.white;
    
    int activeStm = 0, activeNstm = 0;
    i64 screlu_sum_stm = 0, screlu_sum_nstm = 0;
    for (size_t i = 0; i < HL_SIZE; i++) {
        i32 s1 = SCReLU(stm.values[i]);
        i32 s2 = SCReLU(nstm.values[i]);
        if (s1 > 0) activeStm++;
        if (s2 > 0) activeNstm++;
        screlu_sum_stm += s1;
        screlu_sum_nstm += s2;
    }
    std::cout << "Active neurons STM: " << activeStm << "/" << HL_SIZE << std::endl;
    std::cout << "Active neurons NSTM: " << activeNstm << "/" << HL_SIZE << std::endl;
    std::cout << "SCReLU sum STM: " << screlu_sum_stm << std::endl;
    std::cout << "SCReLU sum NSTM: " << screlu_sum_nstm << std::endl;
    
    std::cout << "\n--- STM vs NSTM Breakdown ---" << std::endl;
    debugVectorizedSCReLU(stm, nstm, bucket);
    
    i32 rawEval = vectorizedSCReLU(stm, nstm, bucket);
    std::cout << "\nRaw vectorizedSCReLU result: " << rawEval << std::endl;
    
    i64 afterDivQA = rawEval / QA;
    std::cout << "After /QA: " << afterDivQA << std::endl;
    
    i64 afterBias = afterDivQA + outputBias[bucket];
    std::cout << "After +bias[" << bucket << "]=" << outputBias[bucket] << ": " << afterBias << std::endl;
    
    int finalEval = static_cast<int>((afterBias * EVAL_SCALE) / (static_cast<i64>(QA) * QB));
    std::cout << "Final eval (*" << EVAL_SCALE << "/" << (QA * QB) << "): " << finalEval << std::endl;
    
    std::cout << "\n--- All Buckets Comparison ---" << std::endl;
    for (usize b = 0; b < OUTPUT_BUCKETS; b++) {
        i32 eval = vectorizedSCReLU(stm, nstm, b);
        i64 evalScaled = eval / QA;
        evalScaled += outputBias[b];
        int cp = static_cast<int>((evalScaled * EVAL_SCALE) / (static_cast<i64>(QA) * QB));
        std::cout << "  Bucket " << b << ": " << cp << " cp" << (b == bucket ? " <-- active" : "") << std::endl;
    }
    
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