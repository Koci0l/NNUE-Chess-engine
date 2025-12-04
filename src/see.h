#pragma once

#include "chess.hpp"

namespace chess::see {
    int value(PieceType pt);
    int gain(const Board& board, const Move& move);
    bool see_ge(const Board& board, const Move& move, int threshold);
}