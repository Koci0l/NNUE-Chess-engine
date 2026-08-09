#pragma once
#include <cstdint>
#include <cstddef>

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using usize = size_t;

enum ActivationFn { ReLU, CReLU, SCRELU };
constexpr ActivationFn ACTIVATION = SCRELU;

constexpr usize HL_SIZE        = 1024;
constexpr i16   QA             = 255;
constexpr i16   QB             = 64;
constexpr int   EVAL_SCALE     = 400;
constexpr usize OUTPUT_BUCKETS = 8;

// ============================================================================
// Input Buckets Configuration (Mirrored)
// ============================================================================
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

constexpr usize NUM_INPUT_BUCKETS = 4; // Max value in BUCKET_LAYOUT + 1
constexpr usize INPUT_SIZE = 768 * NUM_INPUT_BUCKETS;

namespace Search {
    constexpr int TB_MATE_IN_MAX_PLY = 32000;
    constexpr int TB_MATED_IN_MAX_PLY = -32000;
}