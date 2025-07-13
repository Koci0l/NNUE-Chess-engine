#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include "chess.hpp"
#include "nnue.h"

constexpr int MATE_SCORE = 30000;
constexpr int SEARCH_DEPTH = 5;

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


int alphaBeta(chess::Board& board, int depth, int alpha, int beta, ThreadInfo& thread) {
    if (depth == 0) {
        return g_nnue.evaluate(board, thread);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        if (board.inCheck()) {
            return -MATE_SCORE - depth;
        }
        return 0;
    }

    for (const auto& move : moves) {
        const auto us = board.sideToMove();
        const auto captured = (move.typeOf() == chess::Move::ENPASSANT)
            ? chess::Piece(chess::PieceType::PAWN, ~us)
            : board.at(move.to());

        board.makeMove(move);

        int eval = -alphaBeta(board, depth - 1, -beta, -alpha, thread);

        board.unmakeMove(move);

        if (eval >= beta) {
            return beta;
        }
        alpha = std::max(alpha, eval);
    }

    return alpha;
}

chess::Move search(chess::Board& board, int depth, ThreadInfo& thread) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    chess::Move best_move;
    int alpha = -MATE_SCORE - 1;
    int beta = MATE_SCORE + 1;

    if (!moves.empty()) {
        best_move = moves[0];
    }

    for (const auto& move : moves) {
        const auto us = board.sideToMove();
        const auto captured = (move.typeOf() == chess::Move::ENPASSANT)
            ? chess::Piece(chess::PieceType::PAWN, ~us)
            : board.at(move.to());

        board.makeMove(move);
        
        int eval = -alphaBeta(board, depth - 1, -beta, -alpha, thread);
        
        board.unmakeMove(move);

        if (eval > alpha) {
            alpha = eval;
            best_move = move;
        }
    }

    std::cout << "info score cp " << alpha << " depth " << depth << std::endl;

    return best_move;
}

void uci_loop() {
    chess::Board board;
    ThreadInfo thread;

    g_nnue.loadNetwork("quantised.bin");
    board.setFen(chess::constants::STARTPOS);
    thread.accumulatorStack.resetAccumulators(board);

    std::string line;
    while (std::getline(std::cin, line)) {
        auto tokens = split(line, ' ');
        if (tokens.empty()) continue;

        if (tokens[0] == "uci") {
            std::cout << "id name MyNNUEEngine" << std::endl;
            std::cout << "id author Kociolek" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (tokens[0] == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (tokens[0] == "eval") {
            g_nnue.showBuckets(&board, thread.accumulatorStack);
        } else if (tokens[0] == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
            thread.accumulatorStack.resetAccumulators(board);
        } else if (tokens[0] == "position") {
            int move_start_index = -1;
            std::string fen_str;
            if (tokens[1] == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                if (tokens.size() > 2 && tokens[2] == "moves") move_start_index = 3;
            } else if (tokens[1] == "fen") {
                int i = 2;
                for (; i < tokens.size(); ++i) {
                    if (tokens[i] == "moves") {
                        move_start_index = i + 1;
                        break;
                    }
                    fen_str += tokens[i] + " ";
                }
                board.setFen(fen_str);
            }

            thread.accumulatorStack.resetAccumulators(board); // Reset after setting FEN

            if (move_start_index != -1) {
                for (int i = move_start_index; i < tokens.size(); ++i) {
                    chess::Move move = chess::uci::uciToMove(board, tokens[i]);
                    board.makeMove(move);
                }
            }
        } else if (tokens[0] == "go") {
            chess::Move best_move = search(board, SEARCH_DEPTH, thread);
            std::cout << "bestmove " << chess::uci::moveToUci(best_move) << std::endl;

        } else if (tokens[0] == "quit") {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    uci_loop();
    return 0;
}