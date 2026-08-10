#pragma once

#include <cstdint>
#include <cstddef>
#include <array>

using i8  = std::int8_t;
using u8  = std::uint8_t;
using i16 = std::int16_t;
using u16 = std::uint16_t;
using i32 = std::int32_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;
using u64 = std::uint64_t;
using usize = std::size_t;

// Network Architecture Hyperparameters
constexpr usize HL_SIZE = 1024;
constexpr usize INPUT_BUCKETS = 4;
constexpr usize FEATURES_PER_BUCKET = 768;
constexpr usize INPUT_SIZE = FEATURES_PER_BUCKET * INPUT_BUCKETS; // 3072
constexpr usize OUTPUT_BUCKETS = 8;

constexpr i32 QA = 255;
constexpr i32 QB = 64;
constexpr i32 EVAL_SCALE = 400;

constexpr int KING_BUCKET_LAYOUT[32] = {
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
};