#include "see.h"
#include <array>

namespace chess::see {

int value(PieceType pt) {
    static const int values[] = {100, 320, 330, 500, 900, 20000};
    int idx = static_cast<int>(pt);
    return (idx >= 0 && idx < 6) ? values[idx] : 0;
}

int gain(const Board& board, const Move& move) {
    if (move.typeOf() == Move::CASTLING) return 0;
    if (move.typeOf() == Move::ENPASSANT) return value(PieceType::PAWN);
    int score = board.at(move.to()) != Piece::NONE ? value(board.at(move.to()).type()) : 0;
    if (move.typeOf() == Move::PROMOTION) {
        score += value(move.promotionType()) - value(PieceType::PAWN);
    }
    return score;
}

namespace {
    using namespace chess;

    constexpr std::array<PieceType, 6> lvaOrder = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING
    };

    inline PieceType popLeastValuable(const Board& board, Bitboard& occ,
                                      Bitboard& attackers, Color color) {
        attackers &= occ;
        for (auto pt : lvaOrder) {
            Bitboard piece_bb = board.pieces(pt, color);
            Bitboard candidates = attackers & piece_bb & occ;
            if (candidates.count() > 0) {
                Square from = candidates.lsb();
                occ ^= Bitboard::fromSquare(from);
                attackers ^= Bitboard::fromSquare(from);
                return pt;
            }
        }
        return PieceType::NONE;
    }

    inline Bitboard attackersTo(const Board& board, Square sq, Bitboard occ) {
        Bitboard atk(0ULL);
        
        // O(1) Array lookups for Pawns, Knights, Kings
        atk |= attacks::pawn(Color::WHITE, sq) & board.pieces(PieceType::PAWN, Color::BLACK);
        atk |= attacks::pawn(Color::BLACK, sq) & board.pieces(PieceType::PAWN, Color::WHITE);
        atk |= attacks::knight(sq) & (board.pieces(PieceType::KNIGHT, Color::WHITE) |
                                      board.pieces(PieceType::KNIGHT, Color::BLACK));
        atk |= attacks::king(sq) & (board.pieces(PieceType::KING, Color::WHITE) |
                                    board.pieces(PieceType::KING, Color::BLACK));
        
        Bitboard bishopsQueens = board.pieces(PieceType::BISHOP, Color::WHITE) |
                                 board.pieces(PieceType::QUEEN, Color::WHITE) |
                                 board.pieces(PieceType::BISHOP, Color::BLACK) |
                                 board.pieces(PieceType::QUEEN, Color::BLACK);
        // O(1) Magic Bitboard lookups for Bishops
        atk |= attacks::bishop(sq, occ) & bishopsQueens;

        Bitboard rooksQueens = board.pieces(PieceType::ROOK, Color::WHITE) |
                               board.pieces(PieceType::QUEEN, Color::WHITE) |
                               board.pieces(PieceType::ROOK, Color::BLACK) |
                               board.pieces(PieceType::QUEEN, Color::BLACK);
        // O(1) Magic Bitboard lookups for Rooks
        atk |= attacks::rook(sq, occ) & rooksQueens;

        return atk;
    }
}

bool see_ge(const Board& board, const Move& move, int threshold) {
    if (move == Move()) return false;
    if (board.at(move.from()) == Piece::NONE) return false;
    
    int score = gain(board, move) - threshold;
    if (score < 0) return false;
    
    PieceType next = (move.typeOf() == Move::PROMOTION)
        ? move.promotionType()
        : board.at(move.from()).type();
    score -= value(next);
    if (score >= 0) return true;
    
    Square square = move.to();
    Bitboard occupancy = board.occ();
    
    occupancy ^= Bitboard::fromSquare(move.from());
    
    if (move.typeOf() == Move::ENPASSANT) {
        Square epCaptured(move.to().file(), move.from().rank());
        occupancy ^= Bitboard::fromSquare(epCaptured);
        occupancy |= Bitboard::fromSquare(square);
    } else if (board.at(move.to()) == Piece::NONE) {
        occupancy |= Bitboard::fromSquare(square);
    }
    
    Bitboard queens = board.pieces(PieceType::QUEEN, Color::WHITE) |
                      board.pieces(PieceType::QUEEN, Color::BLACK);
    Bitboard bishops = queens | board.pieces(PieceType::BISHOP, Color::WHITE) |
                       board.pieces(PieceType::BISHOP, Color::BLACK);
    Bitboard rooks = queens | board.pieces(PieceType::ROOK, Color::WHITE) |
                     board.pieces(PieceType::ROOK, Color::BLACK);
    
    Bitboard attackers = attackersTo(board, square, occupancy);
    Color us = (board.sideToMove() == Color::WHITE) ? Color::BLACK : Color::WHITE;
    
    while (true) {
        Bitboard ourAttackers = attackers & occupancy;
        if (ourAttackers.count() == 0) break;
        
        next = popLeastValuable(board, occupancy, ourAttackers, us);
        if (next == PieceType::NONE) break;
        
        // 🚀 Update X-Ray attacks using O(1) Magic Bitboards
        if (next == PieceType::PAWN || next == PieceType::BISHOP || next == PieceType::QUEEN) {
            attackers |= attacks::bishop(square, occupancy) & bishops;
        }
        if (next == PieceType::ROOK || next == PieceType::QUEEN) {
            attackers |= attacks::rook(square, occupancy) & rooks;
        }
        
        attackers &= occupancy;
        score = -score - 1 - value(next);
        us = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
        
        if (score >= 0) {
            if (next == PieceType::KING && attackers.count() > 0) {
                us = (us == Color::WHITE) ? Color::BLACK : Color::WHITE;
            }
            break;
        }
    }
    return board.sideToMove() != us;
}

} // namespace chess::see