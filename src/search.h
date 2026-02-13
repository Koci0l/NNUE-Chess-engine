#pragma once

#include "chess.hpp"
#include "types.h"
#include "timeman.h"
#include "nnue.h"

void initLMR();

int quiescence(chess::Board& board, int alpha, int beta, 
               ThreadInfo& thread, int ply_from_root, SearchStats& stats);

int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int ply_from_root,
              ThreadInfo& thread, const TimeManager* tm, SearchStats& stats, bool allow_null,
              chess::Move previous_move, SearchStack* ss, 
              chess::Move excluded_move = chess::Move());

chess::Move search(chess::Board& board, int max_depth, ThreadInfo& thread, TimeManager& tm,
                   int64_t node_limit = 0);

std::vector<chess::Move> extractPV(chess::Board board, int max_depth);