// Verifies the incremental NNUE accumulators against a full recompute.
//
//   build:
//     g++ -std=c++17 -O2 -DNDEBUG -Isrc \
//         -DKOCIOLEK_POLICY_HL=2048 -DPOLICY_SMALL_HL=64 \
//         tools/acc_selfcheck.cpp \
//         src/nnue.cpp src/history.cpp src/search.cpp src/tt.cpp src/see.cpp \
//         src/movepick.cpp src/zobrist.cpp src/timeman.cpp src/policy.cpp \
//         src/policy_diag.cpp -o acc_selfcheck
//
//   run:
//     ./acc_selfcheck quantised-v14.bin
//
// Plays random games, applying moves exactly the way search.cpp does (including
// the "reset on horizontal-mirroring change" path for king moves across the
// d/e file) and compares the incrementally updated accumulator with one that is
// rebuilt from scratch. Any mismatch prints the move and the FEN.
#include "search.h"
#include "nnue.h"
#include "accumulator.h"
#include "history.h"
#include "tt.h"

#include <iostream>
#include <random>
#include <vector>

static bool isHmChanging(const chess::Board& board, const chess::Move& move) {
    chess::Piece piece = board.at(move.from());
    if (piece.type() == chess::PieceType::KING) {
        chess::Square old_ksq = move.from();
        chess::Square new_ksq = move.to();
        if (move.typeOf() == chess::Move::CASTLING) {
            bool king_side = move.to().index() > move.from().index();
            new_ksq = chess::Square::castling_king_square(king_side, piece.color());
        }
        if (((old_ksq.index() % 8) > 3) != ((new_ksq.index() % 8) > 3)) return true;
    }
    return false;
}

static void applyIncremental(AccumulatorStack& stack, chess::Board& board,
                             const chess::Move& move) {
    stack.push();
    if (isHmChanging(board, move)) {
        board.makeMove(move);
        stack.current().resetAccumulators(board);
    } else {
        updateAccumulatorForMove(stack, board, move);
        board.makeMove(move);
    }
}

static int evalFresh(chess::Board& board) {
    AccumulatorPair fresh;
    fresh.resetAccumulators(board);
    return g_nnue.forwardPass(&board, fresh);
}

int main(int argc, char** argv) {
    g_nnue.loadNetwork(argc > 1 ? argv[1] : "quantised-v14.bin");
    initTT(16);

    std::mt19937_64 rng(12345);
    ThreadInfo thread;
    int mismatches = 0;
    long checked = 0;

    for (int game = 0; game < 60 && mismatches < 20; ++game) {
        chess::Board board;
        board.setFen(chess::constants::STARTPOS);
        thread.accumulatorStack.resetAccumulators(board);

        for (int ply = 0; ply < 120; ++ply) {
            chess::Movelist moves;
            chess::movegen::legalmoves(moves, board);
            if (moves.empty()) break;

            std::uniform_int_distribution<int> d(0, static_cast<int>(moves.size()) - 1);
            chess::Move move = moves[d(rng)];

            applyIncremental(thread.accumulatorStack, board, move);

            const int inc = g_nnue.evaluate(board, thread);
            const int fresh = evalFresh(board);
            ++checked;
            if (inc != fresh) {
                ++mismatches;
                if (mismatches <= 20) {
                    std::cout << "MISMATCH ply " << ply << " move "
                              << chess::uci::moveToUci(move) << " incremental=" << inc
                              << " fresh=" << fresh << "\n  fen=" << board.getFen() << "\n";
                }
            }
            // Keep the line going (unmake half of the plies so we test both the
            // incremental apply and the push/pop path).
            if (ply > 60) {
                board.unmakeMove(move);
                thread.accumulatorStack.pop();
            }
        }
    }

    std::cout << "checked=" << checked << " mismatches=" << mismatches << "\n";
    return mismatches == 0 ? 0 : 1;
}
