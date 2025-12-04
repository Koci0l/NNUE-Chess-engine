#include "movepick.h"
#include "history.h"
#include "see.h"

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move,
                         const MovePickerContext& ctx) {
    if (move == ctx.tt_move && ctx.tt_move != chess::Move())
        return 1000000;

    if (move.typeOf() == chess::Move::PROMOTION) {
        return 900000 + pieceValue(move.promotionType());
    }

    chess::Piece captured = board.at(move.to());
    bool is_tactical = captured != chess::Piece::NONE || 
                       move.typeOf() == chess::Move::ENPASSANT || 
                       move.typeOf() == chess::Move::PROMOTION;

    if (is_tactical) {
        int see_score = chess::see::see_ge(board, move, 0) ? 100 : 
                       (chess::see::see_ge(board, move, -50) ? 0 : -100);
        int victimValue = captured != chess::Piece::NONE ? pieceValue(captured.type()) : 100;
        int attackerValue = pieceValue(board.at(move.from()).type());
        return 800000 + see_score * 1000 + victimValue * 10 - attackerValue;
    }

    if (move.typeOf() == chess::Move::ENPASSANT) {
        return 800000 + 100;
    }

    int killer_score = g_killerMoves.get_killer_score(ctx.ply, move);
    if (killer_score > 0) 
        return 700000 + killer_score * 1000;

    if (move == ctx.counter_move && ctx.counter_move != chess::Move()) 
        return 650000;

    int score = g_butterflyHistory.get(ctx.side_to_move, move.from(), move.to());
    if (move.typeOf() == chess::Move::CASTLING) 
        score += 50;

    return score;
}

std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves, 
                                    const chess::Board& board,
                                    const MovePickerContext& ctx) {
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());
    for (const auto& move : moves)
        scored.emplace_back(move, scoreMoveForOrdering(board, move, ctx));
    return scored;
}

void pickNextMove(std::vector<ScoredMove>& moves, size_t current) {
    if (current >= moves.size()) return;
    size_t best_idx = current;
    int best_score = moves[current].score;
    for (size_t i = current + 1; i < moves.size(); ++i) {
        if (moves[i].score > best_score) {
            best_score = moves[i].score;
            best_idx = i;
        }
    }
    if (best_idx != current)
        std::swap(moves[current], moves[best_idx]);
}