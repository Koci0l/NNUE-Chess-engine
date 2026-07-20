#pragma once

#include "chess.hpp"
#include <cstdint>

void initZobrist();

uint64_t getZobristHash(const chess::Board& board);

uint64_t getPawnHash(const chess::Board& board);
uint64_t getMaterialHash(const chess::Board& board);