#pragma once

#include "config.h"
#include "accumulator.h"
#include "chess.hpp"
#include <atomic>

struct TimeManager;
struct SearchStats;
struct SearchStack;

struct SearchStats {
    std::atomic<uint64_t> nodes{0};

    SearchStats() = default;
    SearchStats(const SearchStats& other) : nodes(other.nodes.load(std::memory_order_relaxed)) {}
    SearchStats(SearchStats&& other) noexcept : nodes(other.nodes.load(std::memory_order_relaxed)) {}
    SearchStats& operator=(const SearchStats& other) {
        nodes.store(other.nodes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
    SearchStats& operator=(SearchStats&& other) noexcept {
        nodes.store(other.nodes.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    void reset() { nodes.store(0, std::memory_order_relaxed); }
};

struct ThreadInfo {
    int thread_id = 0;
    AccumulatorStack accumulatorStack;
    SearchStats stats;
    
    void reset(const chess::Board& board) {
        accumulatorStack.resetAccumulators(board);
        stats.reset();
    }
};

constexpr int MATE_SCORE = 30000;
constexpr int MAX_PLY = 100;
constexpr int CONTEMPT = 0;

inline chess::Color oppColor(chess::Color c) {
    return c == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE;
}

inline int pieceValue(chess::PieceType pt) {
    static const int values[] = {100, 320, 330, 500, 900, 20000};
    int idx = static_cast<int>(pt);
    return (idx >= 0 && idx < 6) ? values[idx] : 0;
}

inline bool isQuietMove(const chess::Board& board, const chess::Move& move) {
    return board.at(move.to()) == chess::Piece::NONE &&
           move.typeOf() != chess::Move::PROMOTION &&
           move.typeOf() != chess::Move::ENPASSANT;
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

struct SearchStack {
    int static_eval = 0;
    chess::Move current_move;
    chess::Move excluded_move;
    chess::Piece moved_piece = chess::Piece::NONE;
};