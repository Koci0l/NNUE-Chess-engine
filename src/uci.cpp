#include "uci.h"
#include "search.h"
#include "tt.h"
#include "history.h"
#include "zobrist.h"
#include "nnue.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

// Helper to split string by delimiter
static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

// Helper to check if string is a number
static bool is_integer(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

void uci_loop() {
    chess::Board board;
    ThreadInfo thread;

    // Initialize subsystems
    initZobrist();
    initLMR();
    initTT(256); // Default 256MB hash

    std::cout << "info string Loading NNUE..." << std::endl;
    g_nnue.loadNetwork("quantised-v6.bin"); 
    std::cout << "info string NNUE loaded" << std::endl;

    // Set initial state
    board.setFen(chess::constants::STARTPOS);
    thread.accumulatorStack.resetAccumulators(board);

    std::string line;
    while (std::getline(std::cin, line)) {
        auto tokens = split(line, ' ');
        if (tokens.empty()) continue;
        
        std::string command = tokens[0];

        if (command == "uci") {
            std::cout << "id name MyNNUEEngine v20.0-ProbCut" << std::endl;
            std::cout << "id author Kociolek" << std::endl;
            std::cout << "option name Hash type spin default 256 min 1 max 1024" << std::endl;
            std::cout << "uciok" << std::endl;
            std::cout.flush();

        } else if (command == "setoption") {
            if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "Hash" && tokens[3] == "value") {
                int mb = std::stoi(tokens[4]);
                initTT(mb);
            }

        } else if (command == "isready") {
            std::cout << "readyok" << std::endl;
            std::cout.flush();

        } else if (command == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
            thread.accumulatorStack.resetAccumulators(board);
            clearTT();
            g_butterflyHistory.clear();
            g_killerMoves.clear();
            g_counterMoves.clear();
            g_captureHistory.clear();

        } else if (command == "position") {
            size_t moves_idx = 0;
            if (tokens.size() > 1 && tokens[1] == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                moves_idx = 2;
            } else if (tokens.size() > 1 && tokens[1] == "fen") {
                std::string fen;
                size_t i = 2;
                while (i < tokens.size() && tokens[i] != "moves") {
                    fen += tokens[i] + " ";
                    i++;
                }
                board.setFen(fen);
                moves_idx = i;
            }

            if (moves_idx < tokens.size() && tokens[moves_idx] == "moves") {
                for (size_t i = moves_idx + 1; i < tokens.size(); ++i) {
                    chess::Move move = chess::uci::uciToMove(board, tokens[i]);
                    if (move != chess::Move()) {
                        board.makeMove(move);
                    }
                }
            }
            thread.accumulatorStack.resetAccumulators(board);

        } else if (command == "go") {
            int wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 30;
            int depth = 99;
            int movetime = -1;

            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "wtime" && i + 1 < tokens.size()) wtime = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "btime" && i + 1 < tokens.size()) btime = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "winc" && i + 1 < tokens.size()) winc = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "binc" && i + 1 < tokens.size()) binc = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "depth" && i + 1 < tokens.size()) depth = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "movetime" && i + 1 < tokens.size()) movetime = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "movestogo" && i + 1 < tokens.size()) movestogo = std::stoi(tokens[i + 1]);
            }

            if (tokens.size() == 2 && is_integer(tokens[1])) {
                depth = std::stoi(tokens[1]);
            }

            TimeManager tm;
            int time_left = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
            int inc = (board.sideToMove() == chess::Color::WHITE) ? winc : binc;
            
            tm.init(time_left, inc, movestogo, movetime);

            chess::Move best = search(board, depth, thread, tm);
            std::cout << "bestmove " << chess::uci::moveToUci(best) << std::endl;
            std::cout.flush();

        } else if (command == "quit") {
            break;
        
        } else if (command == "eval") {
            int val = g_nnue.evaluate(board, thread);
            std::cout << "NNUE static eval: " << val << std::endl;

        } else if (command == "debug") {
            // Full NNUE debug output
            thread.accumulatorStack.resetAccumulators(board);
            g_nnue.debugNetwork(board, thread.accumulatorStack.current());

        } else if (command == "buckets") {
            // Show all bucket evaluations
            thread.accumulatorStack.resetAccumulators(board);
            g_nnue.showBuckets(&board, thread.accumulatorStack.current());

        } else if (command == "d" || command == "display") {
            // Display current board
            std::cout << board << std::endl;
            std::cout << "FEN: " << board.getFen() << std::endl;
            std::cout << "Side to move: " << (board.sideToMove() == chess::Color::WHITE ? "White" : "Black") << std::endl;
            std::cout << "Pieces: " << board.occ().count() << std::endl;
        }
    }

    delete[] tt;
}