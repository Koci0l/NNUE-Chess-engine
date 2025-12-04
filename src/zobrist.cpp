#include "zobrist.h"
#include <random>

static uint64_t zobrist_piece[12][64];
static uint64_t zobrist_side;
static uint64_t zobrist_castling[16];
static uint64_t zobrist_ep_file[8];

void initZobrist() {
    std::mt19937_64 rng(0x1337BEEF);
    for (int p = 0; p < 12; ++p)
        for (int sq = 0; sq < 64; ++sq)
            zobrist_piece[p][sq] = rng();
    zobrist_side = rng();
    for (int i = 0; i < 16; ++i) zobrist_castling[i] = rng();
    for (int f = 0; f < 8; ++f) zobrist_ep_file[f] = rng();
}

uint64_t getZobristHash(const chess::Board& board) {
    uint64_t hash = 0;
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece p = board.at(chess::Square(sq));
        if (p != chess::Piece::NONE) {
            int piece_idx = static_cast<int>(p.type()) + 
                           (p.color() == chess::Color::WHITE ? 0 : 6);
            hash ^= zobrist_piece[piece_idx][sq];
        }
    }
    if (board.sideToMove() == chess::Color::BLACK) hash ^= zobrist_side;

    int castling_rights = 0;
    if (board.castlingRights().has(chess::Color::WHITE, 
        chess::Board::CastlingRights::Side::KING_SIDE)) castling_rights |= 1;
    if (board.castlingRights().has(chess::Color::WHITE, 
        chess::Board::CastlingRights::Side::QUEEN_SIDE)) castling_rights |= 2;
    if (board.castlingRights().has(chess::Color::BLACK, 
        chess::Board::CastlingRights::Side::KING_SIDE)) castling_rights |= 4;
    if (board.castlingRights().has(chess::Color::BLACK, 
        chess::Board::CastlingRights::Side::QUEEN_SIDE)) castling_rights |= 8;
    hash ^= zobrist_castling[castling_rights];

    if (board.enpassantSq() != chess::Square::underlying::NO_SQ) {
        hash ^= zobrist_ep_file[static_cast<int>(board.enpassantSq().file())];
    }
    return hash;
}