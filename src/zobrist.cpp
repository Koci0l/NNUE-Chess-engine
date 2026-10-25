#include "zobrist.h"

    void initZobrist() {

    }

uint64_t getZobristHash(const chess::Board& board) {
    // board.zobrist() recomputes the whole key from scratch (piece loop + magic
    // lookups) and is called at every search node. board.hash() returns the key
    // the board already maintains incrementally while making/unmaking moves -
    // same value, a few hundred times cheaper.
    return board.hash();
}

    uint64_t getPawnHash(const chess::Board& board) {
    uint64_t wp = board.pieces(chess::PieceType::PAWN, chess::Color::WHITE).getBits();
    uint64_t bp = board.pieces(chess::PieceType::PAWN, chess::Color::BLACK).getBits();
    return wp ^ (bp * 0x9E3779B97F4A7C15ULL);
}

uint64_t getMaterialHash(const chess::Board& board) {
    uint64_t h = 0;
    const chess::PieceType pts[] = {
        chess::PieceType::PAWN,   chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK,   chess::PieceType::QUEEN,  chess::PieceType::KING
    };
    for (int i = 0; i < 6; ++i) {
        int wc = board.pieces(pts[i], chess::Color::WHITE).count();
        int bc = board.pieces(pts[i], chess::Color::BLACK).count();
        h = h * 31 + uint64_t(wc) * 7 + uint64_t(bc);
    }
    return h;
}