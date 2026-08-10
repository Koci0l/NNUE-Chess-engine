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
    if (x > QA) return static_cast<i16>(QA);
    return x;
}

i32 NNUE::SCReLU(const i16 x) {
    const i32 clamped = std::clamp(static_cast<i32>(x), 0, static_cast<i32>(QA));
    return clamped * clamped;
}

// ============================================================================
// Input bucket / mirror helpers  (ChessBucketsMirrored)
// ============================================================================

bool NNUE::kingMirror(chess::Color perspective, chess::Square kingSq) {
    int sq = kingSq.index();
    if (perspective == chess::Color::BLACK) sq ^= 56;
    return (sq & 7) >= 4;
}

int NNUE::kingBucket(chess::Color perspective, chess::Square kingSq) {
    int sq = kingSq.index();
    if (perspective == chess::Color::BLACK) sq ^= 56;

    // Mirror horizontally onto files a–d
    if ((sq & 7) >= 4) sq ^= 7;

    const int idx = ((sq >> 3) << 2) | (sq & 7);
    return KING_BUCKET_LAYOUT[idx];
}

// ============================================================================
// Material Bucket  (MaterialCount::<8>)
// bullet: divisor = 32.div_ceil(N); bucket = (occ - 2) / divisor
// ============================================================================

usize NNUE::getMaterialBucket(const chess::Board& board) {
    constexpr usize divisor = (32 + OUTPUT_BUCKETS - 1) / OUTPUT_BUCKETS;
    const int pieceCount = board.occ().count();
    return static_cast<usize>((pieceCount - 2) / divisor);
}

// ============================================================================
// Feature Index
//   bucket * 768 + colorIndex * 384 + pieceType * 64 + square
// ============================================================================

usize NNUE::feature(chess::Color perspective,
                    chess::Square kingSq,
                    chess::Color color,
                    chess::PieceType piece,
                    chess::Square square) {
    int ksq = kingSq.index();
    int psq = square.index();

    if (perspective == chess::Color::BLACK) {
        ksq ^= 56;
        psq ^= 56;
    }

    if ((ksq & 7) >= 4) {
        ksq ^= 7;
        psq ^= 7;
    }

    const int bucket = KING_BUCKET_LAYOUT[((ksq >> 3) << 2) | (ksq & 7)];
    const int colorIndex = (perspective == color) ? 0 : 1;
    const int pieceIndex = static_cast<int>(piece);

    return static_cast<usize>(
        bucket * 768 + colorIndex * 384 + pieceIndex * 64 + psq
    );
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

    const Vectori16 VEC_QA   = vec_set1_epi16(static_cast<i16>(QA));
    const Vectori16 VEC_ZERO = vec_set1_epi16(0);

    Vectori32 accumulator = vec_setzero_epi32();

    for (usize i = 0; i < HL_SIZE; i += VECTOR_SIZE) {
        const Vectori16 stmValues  = vec_load_epi16(&stm.values[i]);
        const Vectori16 nstmValues = vec_load_epi16(&nstm.values[i]);

        const Vectori16 stmClamped  = vec_min_epi16(VEC_QA, vec_max_epi16(stmValues,  VEC_ZERO));
        const Vectori16 nstmClamped = vec_min_epi16(VEC_QA, vec_max_epi16(nstmValues, VEC_ZERO));

        const Vectori16 stmWeights  = vec_load_epi16(&weightsToOut[bucket][i]);
        const Vectori16 nstmWeights = vec_load_epi16(&weightsToOut[bucket][i + HL_SIZE]);

        const Vectori32 stmActivated  = vec_madd_epi16(stmClamped,  vec_mullo_epi16(stmClamped,  stmWeights));
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
        res += SCReLU(stm.values[i])  * weightsToOut[bucket][i];
        res += SCReLU(nstm.values[i]) * weightsToOut[bucket][i + HL_SIZE];
    }
    return res;
}

#endif

// ============================================================================
// Network Loading
// ============================================================================

void NNUE::loadNetwork(const std::string& filepath) {
    std::ifstream stream(filepath, std::ios::binary);
    if (!stream.is_open()) {
        std::cerr << "ERROR: Failed to open network file: " << filepath << std::endl;
        return;
    }

    stream.seekg(0, std::ios::end);
    const size_t fileSize = static_cast<size_t>(stream.tellg());
    stream.seekg(0, std::ios::beg);

    const size_t expectedSize =
        sizeof(i16) * (INPUT_SIZE * HL_SIZE + HL_SIZE + 2 * HL_SIZE * OUTPUT_BUCKETS + OUTPUT_BUCKETS);

    std::cout << "Network file size: " << fileSize << " bytes" << std::endl;
    std::cout << "Expected size:     " << expectedSize << " bytes" << std::endl;
    std::cout << "  HL_SIZE=" << HL_SIZE
              << " INPUT_BUCKETS=" << INPUT_BUCKETS
              << " INPUT_SIZE=" << INPUT_SIZE
              << " OUTPUT_BUCKETS=" << OUTPUT_BUCKETS
              << " QA=" << QA << " QB=" << QB << std::endl;

    if (fileSize != expectedSize) {
        std::cerr << "WARNING: file size mismatch — wrong net or stale constants?" << std::endl;
    }

    stream.read(reinterpret_cast<char*>(weightsToHL.data()),
                static_cast<std::streamsize>(weightsToHL.size() * sizeof(i16)));
    std::cout << "After weightsToHL, position: " << stream.tellg() << std::endl;

    stream.read(reinterpret_cast<char*>(hiddenLayerBias.data()),
                static_cast<std::streamsize>(hiddenLayerBias.size() * sizeof(i16)));
    std::cout << "After hiddenLayerBias, position: " << stream.tellg() << std::endl;

    for (usize bucket = 0; bucket < OUTPUT_BUCKETS; ++bucket) {
        stream.read(reinterpret_cast<char*>(weightsToOut[bucket].data()),
                    static_cast<std::streamsize>(weightsToOut[bucket].size() * sizeof(i16)));
    }
    std::cout << "After weightsToOut, position: " << stream.tellg() << std::endl;

    stream.read(reinterpret_cast<char*>(outputBias.data()),
                static_cast<std::streamsize>(outputBias.size() * sizeof(i16)));
    std::cout << "After outputBias, position: " << stream.tellg() << std::endl;

    if (!stream) {
        std::cerr << "ERROR: Malformed or incomplete network file: " << filepath << std::endl;
    } else {
        std::cout << "NNUE file loaded successfully: " << filepath << std::endl;
        std::cout << "  Output biases: ";
        for (size_t i = 0; i < OUTPUT_BUCKETS; i++) std::cout << outputBias[i] << " ";
        std::cout << std::endl;
    }
}

// ============================================================================
// Forward Pass
// ============================================================================

int NNUE::forwardPass(const chess::Board* board, const AccumulatorPair& accumulators) {
    const usize outputBucket = getMaterialBucket(*board);
    const bool isWhiteSTM = board->sideToMove() == chess::Color::WHITE;

    const Accumulator& accumulatorSTM  = isWhiteSTM ? accumulators.white : accumulators.black;
    const Accumulator& accumulatorNSTM = isWhiteSTM ? accumulators.black : accumulators.white;

    i64 eval = vectorizedSCReLU(accumulatorSTM, accumulatorNSTM, outputBucket);

    eval /= QA;
    eval += outputBias[outputBucket];

    return static_cast<int>((eval * EVAL_SCALE) / (static_cast<i64>(QA) * QB));
}

i16 NNUE::evaluate(const chess::Board& board, ThreadInfo& thisThread) {
    const int eval = g_nnue.forwardPass(&board, thisThread.accumulatorStack.current());
    // Keep out of mate range (MATE_SCORE from types.h)
    return static_cast<i16>(std::clamp(eval, -MATE_SCORE + 100, MATE_SCORE - 100));
}

// ============================================================================
// Accumulator Updates
// ============================================================================

void AccumulatorPair::add_piece(const chess::Piece& p, const chess::Square& sq) {
    if (p == chess::Piece::NONE) return;

    const chess::Square wk(whiteKing);
    const chess::Square bk(blackKing);

    const usize featureW = NNUE::feature(chess::Color::WHITE, wk, p.color(), p.type(), sq);
    const usize featureB = NNUE::feature(chess::Color::BLACK, bk, p.color(), p.type(), sq);

    const i16* wRow = &g_nnue.weightsToHL[featureW * HL_SIZE];
    const i16* bRow = &g_nnue.weightsToHL[featureB * HL_SIZE];

    for (usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] = static_cast<i16>(white.values[i] + wRow[i]);
        black.values[i] = static_cast<i16>(black.values[i] + bRow[i]);
    }
}

void AccumulatorPair::remove_piece(const chess::Piece& p, const chess::Square& sq) {
    if (p == chess::Piece::NONE) return;

    const chess::Square wk(whiteKing);
    const chess::Square bk(blackKing);

    const usize featureW = NNUE::feature(chess::Color::WHITE, wk, p.color(), p.type(), sq);
    const usize featureB = NNUE::feature(chess::Color::BLACK, bk, p.color(), p.type(), sq);

    const i16* wRow = &g_nnue.weightsToHL[featureW * HL_SIZE];
    const i16* bRow = &g_nnue.weightsToHL[featureB * HL_SIZE];

    for (usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] = static_cast<i16>(white.values[i] - wRow[i]);
        black.values[i] = static_cast<i16>(black.values[i] - bRow[i]);
    }
}

void AccumulatorPair::move_piece(const chess::Piece& p,
                                 const chess::Square& from,
                                 const chess::Square& to) {
    if (p == chess::Piece::NONE) return;

    const chess::Square wk(whiteKing);
    const chess::Square bk(blackKing);

    const usize featureFromW = NNUE::feature(chess::Color::WHITE, wk, p.color(), p.type(), from);
    const usize featureFromB = NNUE::feature(chess::Color::BLACK, bk, p.color(), p.type(), from);
    const usize featureToW   = NNUE::feature(chess::Color::WHITE, wk, p.color(), p.type(), to);
    const usize featureToB   = NNUE::feature(chess::Color::BLACK, bk, p.color(), p.type(), to);

    const i16* wFrom = &g_nnue.weightsToHL[featureFromW * HL_SIZE];
    const i16* bFrom = &g_nnue.weightsToHL[featureFromB * HL_SIZE];
    const i16* wTo   = &g_nnue.weightsToHL[featureToW   * HL_SIZE];
    const i16* bTo   = &g_nnue.weightsToHL[featureToB   * HL_SIZE];

    for (usize i = 0; i < HL_SIZE; ++i) {
        white.values[i] = static_cast<i16>(white.values[i] + wTo[i] - wFrom[i]);
        black.values[i] = static_cast<i16>(black.values[i] + bTo[i] - bFrom[i]);
    }
}

bool AccumulatorPair::kingBucketChanged(const chess::Piece& p,
                                        const chess::Square& from,
                                        const chess::Square& to) const {
    if (p.type() != chess::PieceType::KING) return false;

    const chess::Color c = p.color();
    const int  bucketFrom = NNUE::kingBucket(c, from);
    const int  bucketTo   = NNUE::kingBucket(c, to);
    const bool mirrorFrom = NNUE::kingMirror(c, from);
    const bool mirrorTo   = NNUE::kingMirror(c, to);
    return bucketFrom != bucketTo || mirrorFrom != mirrorTo;
}

void AccumulatorPair::resetAccumulators(const chess::Board& board) {
    std::memcpy(white.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));
    std::memcpy(black.values, g_nnue.hiddenLayerBias.data(), HL_SIZE * sizeof(i16));

    whiteKing = static_cast<uint8_t>(board.kingSq(chess::Color::WHITE).index());
    blackKing = static_cast<uint8_t>(board.kingSq(chess::Color::BLACK).index());

    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        chess::Square sq(sq_idx);
        chess::Piece p = board.at(sq);
        if (p != chess::Piece::NONE) {
            add_piece(p, sq);
        }
    }
}

// ============================================================================
// Debug
// ============================================================================

void NNUE::debugVectorizedSCReLU(const Accumulator& stm, const Accumulator& nstm, usize bucket) {
    i64 stm_contrib = 0, nstm_contrib = 0;

    for (usize i = 0; i < HL_SIZE; i++) {
        stm_contrib  += static_cast<i64>(SCReLU(stm.values[i]))  * weightsToOut[bucket][i];
        nstm_contrib += static_cast<i64>(SCReLU(nstm.values[i])) * weightsToOut[bucket][i + HL_SIZE];
    }

    std::cout << "STM contribution: "  << stm_contrib  << std::endl;
    std::cout << "NSTM contribution: " << nstm_contrib << std::endl;
    std::cout << "Total: " << (stm_contrib + nstm_contrib) << std::endl;

    i64 stm_cp  = (stm_contrib  * EVAL_SCALE) / (static_cast<i64>(QA) * QA * QB);
    i64 nstm_cp = (nstm_contrib * EVAL_SCALE) / (static_cast<i64>(QA) * QA * QB);
    std::cout << "STM contribution (cp): "  << stm_cp  << std::endl;
    std::cout << "NSTM contribution (cp): " << nstm_cp << std::endl;
}

void NNUE::debugNetwork(const chess::Board& board, const AccumulatorPair& accumulators) {
    std::cout << "\n========== NNUE DEBUG ==========\n" << std::endl;
    std::cout << "HL_SIZE=" << HL_SIZE
              << " INPUT_BUCKETS=" << INPUT_BUCKETS
              << " QA=" << QA << " QB=" << QB
              << " EVAL_SCALE=" << EVAL_SCALE << std::endl;

    const chess::Square wk(accumulators.whiteKing);
    const chess::Square bk(accumulators.blackKing);

    std::cout << "White king: " << int(accumulators.whiteKing)
              << " bucket=" << kingBucket(chess::Color::WHITE, wk)
              << " mirror=" << kingMirror(chess::Color::WHITE, wk) << std::endl;
    std::cout << "Black king: " << int(accumulators.blackKing)
              << " bucket=" << kingBucket(chess::Color::BLACK, bk)
              << " mirror=" << kingMirror(chess::Color::BLACK, bk) << std::endl;

    std::cout << "outputBias: ";
    for (size_t i = 0; i < OUTPUT_BUCKETS; i++) std::cout << outputBias[i] << " ";
    std::cout << std::endl;

    const usize bucket = getMaterialBucket(board);
    const bool isWhiteSTM = board.sideToMove() == chess::Color::WHITE;
    const Accumulator& stm  = isWhiteSTM ? accumulators.white : accumulators.black;
    const Accumulator& nstm = isWhiteSTM ? accumulators.black : accumulators.white;

    i32 rawEval = vectorizedSCReLU(stm, nstm, bucket);
    i64 afterDivQA = rawEval / QA;
    i64 afterBias  = afterDivQA + outputBias[bucket];
    int finalEval  = static_cast<int>((afterBias * EVAL_SCALE) / (static_cast<i64>(QA) * QB));

    std::cout << "Material bucket: " << bucket
              << " pieces: " << board.occ().count() << std::endl;
    std::cout << "Raw SCReLU: " << rawEval
              << " final: " << finalEval << " cp" << std::endl;
    std::cout << "\n================================\n" << std::endl;
}

void NNUE::showBuckets(const chess::Board* board, const AccumulatorPair& accumulators) {
    std::cout << "+------------+------------+\n"
              << "|   Bucket   | Evaluation |\n"
              << "+------------+------------+" << std::endl;

    const usize currentBucket = getMaterialBucket(*board);
    const bool isWhiteSTM = board->sideToMove() == chess::Color::WHITE;
    const Accumulator& accumulatorSTM  = isWhiteSTM ? accumulators.white : accumulators.black;
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