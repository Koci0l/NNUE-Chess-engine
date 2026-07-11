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
#include <chrono>

static const int BENCH_DEPTH = 12;

static const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "2r5/3pk3/8/2P5/8/2K5/8/8 w - - 0 1",
    "rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R b KQkq - 0 6",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "3r3k/2r4p/1p1b3q/p4P2/P2Pp3/1B2P3/3BQ1RP/6K1 w - - 0 1",
    "2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1",
    "r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1",
    "r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1",
    "r3kb1r/p3pp1p/bpNq1np1/8/8/5N2/PP2PPPP/R1BQKB1R b KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "8/8/1P6/5pr1/8/4R3/7k/2K5 w - - 0 1",
    "8/8/8/8/5kp1/P7/8/1K1N4 w - - 0 1",
    "8/8/8/5N2/8/p7/8/2NK3k w - - 0 1",
    "8/8/8/8/8/6k1/6p1/6K1 w - - 0 1",
    "r5k1/pp1Rn1pp/1p2b3/8/2P5/4B3/PPP2PPP/2K5 w - - 0 1",
    "6k1/6p1/6Pp/ppp5/3pn2P/1P3K2/1PP2P2/3N4 b - - 0 1",
};

static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

static void run_bench(ThreadInfo& thread) {
    chess::Board bench_board;

    clearTT();
    g_butterflyHistory.clear();
    g_killerMoves.clear();
    g_counterMoves.clear();
    g_captureHistory.clear();
    g_contHist1ply.clear();
    g_contHist2ply.clear();

    g_silent = true;
    uint64_t total_nodes = 0;
    auto start = std::chrono::high_resolution_clock::now();

    int count = sizeof(BENCH_FENS) / sizeof(BENCH_FENS[0]);
    for (int i = 0; i < count; ++i) {
        bench_board.setFen(BENCH_FENS[i]);
        thread.accumulatorStack.resetAccumulators(bench_board);

        TimeManager tm;
        tm.init(0, 0, 0, 0, 0);

        uint64_t nodes = 0;
        search(bench_board, BENCH_DEPTH, thread, tm, 0, nullptr, &nodes);
        total_nodes += nodes;
    }

    auto end = std::chrono::high_resolution_clock::now();
    g_silent = false;

    int64_t elapsed_ms = std::max<int64_t>(1,
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    uint64_t nps = (total_nodes * 1000) / elapsed_ms;

    std::cout << total_nodes << " nodes " << nps << " nps" << std::endl;
    std::cout.flush();
}

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

static bool process_command(const std::string& line, chess::Board& board, ThreadInfo& thread) {
        auto tokens = split(line, ' ');
        if (tokens.empty()) return true;
        
        std::string command = tokens[0];

        if (command == "uci") {
            std::cout << "id name Kociolek-2.1" << std::endl;
            std::cout << "id author Kociolek" << std::endl;
            std::cout << "option name Hash type spin default 256 min 1 max 1024" << std::endl;
            std::cout << "option name Threads type spin default 1 min 1 max 1" << std::endl;
            std::cout << "uciok" << std::endl;
            std::cout.flush();

        } else if (command == "setoption") {
            if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "Hash" && tokens[3] == "value") {
                int mb = std::stoi(tokens[4]);
                initTT(mb);
            } else if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "EvalFile" && tokens[3] == "value") {
                std::string path;
                for (size_t i = 4; i < tokens.size(); ++i) {
                    path += tokens[i];
                    if (i + 1 < tokens.size()) path += " ";
                }
                g_nnue.loadNetwork(path);
                board.setFen(chess::constants::STARTPOS);
                thread.accumulatorStack.resetAccumulators(board);
            }

        } else if (command == "bench") {
            run_bench(thread);

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
            g_contHist1ply.clear();
            g_contHist2ply.clear();

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
            int64_t nodes = 0;

            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "wtime" && i + 1 < tokens.size()) wtime = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "btime" && i + 1 < tokens.size()) btime = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "winc" && i + 1 < tokens.size()) winc = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "binc" && i + 1 < tokens.size()) binc = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "depth" && i + 1 < tokens.size()) depth = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "movetime" && i + 1 < tokens.size()) movetime = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "movestogo" && i + 1 < tokens.size()) movestogo = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "nodes" && i + 1 < tokens.size()) nodes = std::stoll(tokens[i + 1]);
            }

            if (tokens.size() == 2 && is_integer(tokens[1])) {
                depth = std::stoi(tokens[1]);
            }

            TimeManager tm;
            int time_left = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
            int inc = (board.sideToMove() == chess::Color::WHITE) ? winc : binc;
            
            tm.init(time_left, inc, movestogo, movetime, board.fullMoveNumber() * 2 - (board.sideToMove() == chess::Color::WHITE ? 2 : 1));

            chess::Move best = search(board, depth, thread, tm, nodes);
            std::cout << "bestmove " << chess::uci::moveToUci(best) << std::endl;
            std::cout.flush();

        } else if (command == "quit") {
            return false;
        
        } else if (command == "eval") {
            int val = g_nnue.evaluate(board, thread);
            std::cout << "NNUE static eval: " << val << std::endl;

        } else if (command == "debug") {
            thread.accumulatorStack.resetAccumulators(board);
            g_nnue.debugNetwork(board, thread.accumulatorStack.current());

        } else if (command == "buckets") {
            thread.accumulatorStack.resetAccumulators(board);
            g_nnue.showBuckets(&board, thread.accumulatorStack.current());

        } else if (command == "d" || command == "display") {
            std::cout << board << std::endl;
            std::cout << "FEN: " << board.getFen() << std::endl;
            std::cout << "Side to move: " << (board.sideToMove() == chess::Color::WHITE ? "White" : "Black") << std::endl;
            std::cout << "Pieces: " << board.occ().count() << std::endl;
        }

        return true;
}

void uci_loop(int argc, char* argv[]) {
    chess::Board board;
    ThreadInfo thread;

    initZobrist();
    initLMR();
    initTT(256);

    std::cout << "info string Loading NNUE..." << std::endl;
    g_nnue.loadNetwork("(768-1024)x2-1-8.bin");
    std::cout << "info string NNUE loaded" << std::endl;

    board.setFen(chess::constants::STARTPOS);
    thread.accumulatorStack.resetAccumulators(board);

    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if (!process_command(argv[i], board, thread)) break;
        }
        delete[] tt;
        return;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!process_command(line, board, thread)) break;
    }

    delete[] tt;
}