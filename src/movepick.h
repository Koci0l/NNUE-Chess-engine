#pragma once

#include "chess.hpp"
#include "types.h"
#include <vector>
#include "types.h" 

struct MovePickerContext {
    chess::Move tt_move;
    chess::Move counter_move;
    chess::Color side_to_move;
    int ply;
    
    MovePickerContext(chess::Move tt, chess::Move counter, chess::Color stm, int p)
        : tt_move(tt), counter_move(counter), side_to_move(stm), ply(p) {}
};

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move,
                         const MovePickerContext& ctx);
std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves, 
                                    const chess::Board& board,
                                    const MovePickerContext& ctx);
void pickNextMove(std::vector<ScoredMove>& moves, size_t current);