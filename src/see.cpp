#include "see.h"
#include <array>

namespace {
    using namespace chess;

    inline Bitboard pawnAttacks(Color c, Square sq) {
        Bitboard attacks(0ULL);
        int f = static_cast<int>(sq.file());
        int r = static_cast<int>(sq.rank());

        if (c == Color::WHITE) {
            if (f > 0 && r < 7) attacks |= Bitboard::fromSquare(Square(f - 1 + (r + 1) * 8));
            if (f < 7 && r < 7) attacks |= Bitboard::fromSquare(Square(f + 1 + (r + 1) * 8));
        } else {
            if (f > 0 && r > 0) attacks |= Bitboard::fromSquare(Square(f - 1 + (r - 1) * 8));
            if (f < 7 && r > 0) attacks |= Bitboard::fromSquare(Square(f + 1 + (r - 1) * 8));
        }
        return attacks;
    }

    inline Bitboard knightAttacks(Square sq) {
        Bitboard attacks(0ULL);
        int f = static_cast<int>(sq.file());
        int r = static_cast<int>(sq.rank());
        int deltas[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
        for (int i = 0; i < 8; ++i) {
            int nf = f + deltas[i][0], nr = r + deltas[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                attacks |= Bitboard::fromSquare(Square(nf + nr * 8));
            }
        }
        return attacks;
    }

    inline Bitboard kingAttacks(Square sq) {
        Bitboard attacks(0ULL);
        int f = static_cast<int>(sq.file());
        int r = static_cast<int>(sq.rank());
        for (int df = -1; df <= 1; ++df) {
            for (int dr = -1; dr <= 1; ++dr) {
                if (df == 0 && dr == 0) continue;
                int nf = f + df, nr = r + dr;
                if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
                    attacks |= Bitboard::fromSquare(Square(nf + nr * 8));
                }
            }
        }
        return attacks;
    }

    inline Bitboard getBishopAttacks(Square sq, Bitboard occ) {
        Bitboard attacks(0ULL);
        int f = static_cast<int>(sq.file());
        int r = static_cast<int>(sq.rank());
        
        int dirs[4][2] = {{1,1}, {-1,1}, {1,-1}, {-1,-1}};
        for (auto& dir : dirs) {
            for (int d = 1; d < 8; ++d) {
                int nf = f + d * dir[0], nr = r + d * dir[1];
                if (nf < 0 || nf >= 8 || nr < 0 || nr >= 8) break;
                Square nsq(nf + nr * 8);
                attacks |= Bitboard::fromSquare(nsq);
                if (occ & Bitboard::fromSquare(nsq)) break;
            }
        }
        return attacks;
    }

    inline Bitboard getRookAttacks(Square sq, Bitboard occ) {
        Bitboard attacks(0ULL);
        int f = static_cast<int>(sq.file());
        int r = static_cast<int>(sq.rank());

        for (int nr = r + 1; nr < 8; ++nr) {
            Square nsq(f + nr * 8);
            attacks |= Bitboard::fromSquare(nsq);
            if (occ & Bitboard::fromSquare(nsq)) break;
        }
        for (int nr = r - 1; nr >= 0; --nr) {
            Square nsq(f + nr * 8);
            attacks |= Bitboard::fromSquare(nsq);
            if (occ & Bitboard::fromSquare(nsq)) break;
        }
        for (int nf = f + 1; nf < 8; ++nf) {
            Square nsq(nf + r * 8);
            attacks |= Bitboard::fromSquare(nsq);
            if (occ & Bitboard::fromSquare(nsq)) break;
        }
        for (int nf = f - 1; nf >= 0; --nf) {
            Square nsq(nf + r * 8);
            attacks |= Bitboard::fromSquare(nsq);
            if (occ & Bitboard::fromSquare(nsq)) break;
        }
        return attacks;
    }

    constexpr std::array<PieceType, 6> lvaOrder = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING
    };

    inline PieceType popLeastValuable(const Board& board, Bitboard& occ, 
                                       Bitboard& attackers, Color color) {
        for (auto pt : lvaOrder) {
            auto piece_bb = board.pieces(pt, color);
            auto candidates = attackers & piece_bb;
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
        atk |= pawnAttacks(Color::WHITE, sq) & board.pieces(PieceType::PAWN, Color::BLACK);
        atk |= pawnAttacks(Color::BLACK, sq) & board.pieces(PieceType::PAWN, Color::WHITE);
        atk |= knightAttacks(sq) & (board.pieces(PieceType::KNIGHT, Color::WHITE) | 
                                     board.pieces(PieceType::KNIGHT, Color::BLACK));
        atk |= kingAttacks(sq) & (board.pieces(PieceType::KING, Color::WHITE) | 
                                   board.pieces(PieceType::KING, Color::BLACK));

        Bitboard bishopsQueens = board.pieces(PieceType::BISHOP, Color::WHITE) | 
                                  board.pieces(PieceType::QUEEN, Color::WHITE) |
                                  board.pieces(PieceType::BISHOP, Color::BLACK) | 
                                  board.pieces(PieceType::QUEEN, Color::BLACK);
        atk |= getBishopAttacks(sq, occ) & bishopsQueens;

        Bitboard rooksQueens = board.pieces(PieceType::ROOK, Color::WHITE) | 
                                board.pieces(PieceType::QUEEN, Color::WHITE) |
                                board.pieces(PieceType::ROOK, Color::BLACK) | 
                                board.pieces(PieceType::QUEEN, Color::BLACK);
        atk |= getRookAttacks(sq, occ) & rooksQueens;

        return atk;
    }
}

namespace chess::see {

int value(PieceType pt) {
    static const int values[] = {100, 320, 330, 500, 900, 20000};
    int idx = static_cast<int>(pt);
    return (idx >= 0 && idx < 6) ? values[idx] : 0;
}

int gain(const Board& board, const Move& move) {
    if (move.typeOf() == Move::CASTLING) return 0;
    if (move.typeOf() == Move::ENPASSANT) return value(PieceType::PAWN);

    auto score = board.at(move.to()) != Piece::NONE ? value(board.at(move.to()).type()) : 0;
    if (move.typeOf() == Move::PROMOTION) {
        score += value(move.promotionType()) - value(PieceType::PAWN);
    }
    return score;
}

bool see_ge(const Board& board, const Move& move, int threshold) {
    auto score = gain(board, move) - threshold;
    if (score < 0) return false;

    PieceType next = (move.typeOf() == Move::PROMOTION) ? 
                     move.promotionType() : board.at(move.from()).type();
    score -= value(next);
    if (score >= 0) return true;

    Square square = move.to();

    // Fix: Use board.occ() instead of manual loop to avoid type conversion error
    Bitboard occupancy = board.occ();

    occupancy ^= Bitboard::fromSquare(move.from());
    occupancy ^= Bitboard::fromSquare(square);

    Bitboard queens = board.pieces(PieceType::QUEEN, Color::WHITE) | 
                      board.pieces(PieceType::QUEEN, Color::BLACK);
    Bitboard bishops = queens | board.pieces(PieceType::BISHOP, Color::WHITE) | 
                       board.pieces(PieceType::BISHOP, Color::BLACK);
    Bitboard rooks = queens | board.pieces(PieceType::ROOK, Color::WHITE) | 
                     board.pieces(PieceType::ROOK, Color::BLACK);

    Bitboard attackers = attackersTo(board, square, occupancy);
    Color us = (board.sideToMove() == Color::WHITE) ? Color::BLACK : Color::WHITE;

    while (true) {
        Bitboard ourAttackers = attackers;
        if (ourAttackers.count() == 0) break;

        next = popLeastValuable(board, occupancy, ourAttackers, us);
        if (next == PieceType::NONE) break;

        if (next == PieceType::PAWN || next == PieceType::BISHOP || next == PieceType::QUEEN) {
            attackers |= getBishopAttacks(square, occupancy) & bishops;
        }
        if (next == PieceType::ROOK || next == PieceType::QUEEN) {
            attackers |= getRookAttacks(square, occupancy) & rooks;
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