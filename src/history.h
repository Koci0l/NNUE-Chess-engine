#pragma once

#include "chess.hpp"
#include "types.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>

struct ButterflyHistory {
    static constexpr int HISTORY_MAX = 16384;
    static constexpr int HISTORY_GRAVITY = HISTORY_MAX;
    int16_t table[2][64][64];

    ButterflyHistory() { clear(); }
    void clear() { std::memset(table, 0, sizeof(table)); }
    
    int get(chess::Color color, chess::Square from, chess::Square to) const {
        return table[static_cast<int>(color)][from.index()][to.index()];
    }

    void update(chess::Color color, chess::Square from, chess::Square to, int bonus) {
        int c = static_cast<int>(color);
        int f = from.index();
        int t = to.index();
        int cur = table[c][f][t];
        int delta = bonus - (cur * std::abs(bonus)) / HISTORY_GRAVITY;
        int next = std::max(-HISTORY_MAX, std::min(HISTORY_MAX, cur + delta));
        table[c][f][t] = static_cast<int16_t>(next);
    }

    void age();
    bool should_age() const;
};

struct KillerMoves {
    chess::Move killers[MAX_PLY][2];

    KillerMoves() { clear(); }
    void clear();
    void store(int ply, chess::Move move);
    bool is_killer(int ply, chess::Move move) const;
    int get_killer_score(int ply, chess::Move move) const;
};

struct CounterMoveHistory {
    chess::Move table[64][64];

    CounterMoveHistory() { clear(); }
    void clear();
    void update(chess::Move previous_move, chess::Move counter_move);
    chess::Move get(chess::Move previous_move) const;
    bool is_counter(chess::Move previous_move, chess::Move move) const;
};

struct CaptureHistory {
    static constexpr int HISTORY_MAX = 16384;
    static constexpr int HISTORY_GRAVITY = HISTORY_MAX;
    int16_t table[6][64][6];

    CaptureHistory() { clear(); }
    void clear() { std::memset(table, 0, sizeof(table)); }

    int get(int piece, int to, int captured) const {
        if (piece < 0 || piece >= 6 || to < 0 || to >= 64 || captured < 0 || captured >= 6) return 0;
        return table[piece][to][captured];
    }

    void update(int piece, int to, int captured, int bonus) {
        if (piece < 0 || piece >= 6 || to < 0 || to >= 64 || captured < 0 || captured >= 6) return;
        int cur = table[piece][to][captured];
        int delta = bonus - (cur * std::abs(bonus)) / HISTORY_GRAVITY;
        int next = std::max(-HISTORY_MAX, std::min(HISTORY_MAX, cur + delta));
        table[piece][to][captured] = static_cast<int16_t>(next);
    }
};

// ============================================================================
// Continuation History: [prev_piece][prev_to][cur_piece][cur_to]
// ============================================================================

struct ContinuationHistory {
    static constexpr int HISTORY_MAX = 16384;
    static constexpr int HISTORY_GRAVITY = HISTORY_MAX;
    static constexpr int NUM_PIECES = 12;   // 6 types * 2 colors
    static constexpr int NUM_SQUARES = 64;

    int16_t table[NUM_PIECES * NUM_SQUARES][NUM_PIECES * NUM_SQUARES];

    ContinuationHistory() { clear(); }
    void clear() { std::memset(table, 0, sizeof(table)); }

    static int pieceIndex(chess::Piece p) {
        return static_cast<int>(p.type()) * 2 + static_cast<int>(p.color());
    }

    int get(chess::Piece prev_piece, chess::Square prev_to,
            chess::Piece cur_piece, chess::Square cur_to) const {
        if (prev_piece == chess::Piece::NONE || cur_piece == chess::Piece::NONE) return 0;
        int prev_idx = pieceIndex(prev_piece) * NUM_SQUARES + prev_to.index();
        int cur_idx = pieceIndex(cur_piece) * NUM_SQUARES + cur_to.index();
        return table[prev_idx][cur_idx];
    }

    void update(chess::Piece prev_piece, chess::Square prev_to,
                chess::Piece cur_piece, chess::Square cur_to, int bonus) {
        if (prev_piece == chess::Piece::NONE || cur_piece == chess::Piece::NONE) return;
        int prev_idx = pieceIndex(prev_piece) * NUM_SQUARES + prev_to.index();
        int cur_idx = pieceIndex(cur_piece) * NUM_SQUARES + cur_to.index();
        int16_t& entry = table[prev_idx][cur_idx];
        int clamped = std::clamp(bonus, -HISTORY_MAX, HISTORY_MAX);
        int delta = clamped - (entry * std::abs(clamped)) / HISTORY_GRAVITY;
        entry = static_cast<int16_t>(std::clamp(int(entry) + delta, -HISTORY_MAX, HISTORY_MAX));
    }

    void age() {
        for (int i = 0; i < NUM_PIECES * NUM_SQUARES; ++i)
            for (int j = 0; j < NUM_PIECES * NUM_SQUARES; ++j)
                table[i][j] /= 2;
    }
};

// Global instances
extern ButterflyHistory g_butterflyHistory;
extern KillerMoves g_killerMoves;
extern CounterMoveHistory g_counterMoves;
extern CaptureHistory g_captureHistory;
extern ContinuationHistory g_contHist1ply;
extern ContinuationHistory g_contHist2ply;