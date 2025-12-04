#pragma once

#include "chess.hpp"
#include <cstdint>

void initZobrist();
uint64_t getZobristHash(const chess::Board& board);