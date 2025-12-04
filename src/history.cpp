#include "history.h"

ButterflyHistory g_butterflyHistory;
KillerMoves g_killerMoves;
CounterMoveHistory g_counterMoves;

void ButterflyHistory::age() {
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 64; ++i)
            for (int j = 0; j < 64; ++j)
                table[c][i][j] /= 2;
}

bool ButterflyHistory::should_age() const {
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 64; ++i)
            for (int j = 0; j < 64; ++j)
                if (std::abs(table[c][i][j]) > (HISTORY_MAX * 7) / 8)
                    return true;
    return false;
}

void KillerMoves::clear() {
    for (int i = 0; i < MAX_PLY; ++i) {
        killers[i][0] = chess::Move();
        killers[i][1] = chess::Move();
    }
}

void KillerMoves::store(int ply, chess::Move move) {
    if (ply >= MAX_PLY) return;
    if (move == killers[ply][0]) return;
    killers[ply][1] = killers[ply][0];
    killers[ply][0] = move;
}

bool KillerMoves::is_killer(int ply, chess::Move move) const {
    if (ply >= MAX_PLY) return false;
    return move == killers[ply][0] || move == killers[ply][1];
}

int KillerMoves::get_killer_score(int ply, chess::Move move) const {
    if (ply >= MAX_PLY) return 0;
    if (move == killers[ply][0]) return 2;
    if (move == killers[ply][1]) return 1;
    return 0;
}

void CounterMoveHistory::clear() {
    for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j)
            table[i][j] = chess::Move();
}

void CounterMoveHistory::update(chess::Move previous_move, chess::Move counter_move) {
    if (previous_move != chess::Move() && counter_move != chess::Move()) {
        table[previous_move.from().index()][previous_move.to().index()] = counter_move;
    }
}

chess::Move CounterMoveHistory::get(chess::Move previous_move) const {
    if (previous_move == chess::Move()) return chess::Move();
    return table[previous_move.from().index()][previous_move.to().index()];
}

bool CounterMoveHistory::is_counter(chess::Move previous_move, chess::Move move) const {
    if (previous_move == chess::Move() || move == chess::Move()) return false;
    return table[previous_move.from().index()][previous_move.to().index()] == move;
}