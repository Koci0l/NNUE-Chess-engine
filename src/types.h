#pragma once

#include "config.h"
#include "accumulator.h"
#include "chess.hpp"

// Forward declarations
struct TimeManager;
struct SearchStats;
struct SearchStack;

// ThreadInfo definition
struct ThreadInfo {
    AccumulatorStack accumulatorStack;
    
    void reset(const chess::Board& board) {
        accumulatorStack.resetAccumulators(board);
    }
};

// Constants
constexpr int MATE_SCORE = 30000;
constexpr int MAX_PLY = 100;
constexpr int CONTEMPT = 0;

// Utility functions
inline chess::Color oppColor(chess::Color c) {
    return c == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE;
}

inline int pieceValue(chess::PieceType pt) {
    static const int values[] = {100, 320, 330, 500, 900, 20000};
    int idx = static_cast<int>(pt);
    return (idx >= 0 && idx < 6) ? values[idx] : 0;
}

// A move is a "capture" if it takes an enemy piece.  IMPORTANT: castling is
// internally encoded as king-takes-own-rook, so board.at(move.to()) is our own
// rook.  Without the CASTLING check castling was classified as a (losing)
// capture: it got SEE-pruned/ordered together with bad captures, was never
// treated as a quiet move and polluted the capture history.
inline bool isCaptureMove(const chess::Board& board, const chess::Move& move) {
    if (move.typeOf() == chess::Move::CASTLING) return false;
    return board.at(move.to()) != chess::Piece::NONE ||
           move.typeOf() == chess::Move::ENPASSANT;
}

inline bool isPromotionMove(const chess::Move& move) {
    return move.typeOf() == chess::Move::PROMOTION;
}

inline bool isQuietMove(const chess::Board& board, const chess::Move& move) {
    return !isCaptureMove(board, move) && !isPromotionMove(move);
}

inline bool hasNonPawnMaterial(const chess::Board& board) {
    chess::Color side = board.sideToMove();
    return board.pieces(chess::PieceType::KNIGHT, side).count() > 0 ||
           board.pieces(chess::PieceType::BISHOP, side).count() > 0 ||
           board.pieces(chess::PieceType::ROOK, side).count() > 0 ||
           board.pieces(chess::PieceType::QUEEN, side).count() > 0;
}

struct ScoredMove {
    chess::Move move;
    int score = 0;

    ScoredMove() = default;
    ScoredMove(const chess::Move& m, int s) : move(m), score(s) {}
};

struct SearchStats {
    uint64_t nodes = 0;
    bool stopped = false;
    void reset() { 
        nodes = 0; stopped = false; 
    }
};

struct SearchStack {
    int static_eval = 0;
    chess::Move current_move;
    chess::Move excluded_move;
    chess::Piece moved_piece = chess::Piece::NONE;
};