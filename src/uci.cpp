#include "uci.h"
#include "search.h"
#include "tt.h"
#include "history.h"
#include "zobrist.h"
#include "nnue.h"
#include "policy.h"
#include "policy_diag.h"

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <cctype>
#include <algorithm>
#include <fstream>
#include <random>

#ifndef __has_include
#define __has_include(x) 0
#endif

#if __has_include("policy_embed.h")
#include "policy_embed.h"
#define HAVE_POLICY_EMBED 1
#else
#define HAVE_POLICY_EMBED 0
#endif

#if __has_include("policy_embed_small.h")
#include "policy_embed_small.h"
#define HAVE_POLICY_SMALL_EMBED 1
#else
#define HAVE_POLICY_SMALL_EMBED 0
#endif

#ifndef EVALFILE
#define EVALFILE "768-1024x2-1-8.bin"
#endif

#ifndef POLICYFILE
#define POLICYFILE "quantised.bin"
#endif

#ifndef POLICYFILE_SMALL
#define POLICYFILE_SMALL "quantised-64.bin"
#endif

static const int BENCH_DEPTH = 14;
static const int GENFEN_RANDOM_PLIES = 8;
static const int GENFEN_EVAL_LIMIT = 400;

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
    g_correctionHistory.clear();
    g_pawnCorrectionHistory.clear();
    g_materialCorrectionHistory.clear();

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

    std::cout << "info string policy_status "
              << (g_policy.loaded ? "LOADED" : "MISSING")
              << " small_policy_status "
              << (g_policy_small.loaded ? "LOADED" : "MISSING")
              << " use_policy "
              << (g_use_policy ? "true" : "false")
              << std::endl;

    std::cout.flush();
}

static bool is_integer(const std::string& s) {
    if (s.empty()) return false;

    size_t i = 0;

    if (s[0] == '-' || s[0] == '+')
        i = 1;

    if (i >= s.size()) return false;

    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i])))
            return false;
    }

    return true;
}

static std::string trim_ws(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])))
        ++a;

    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
        --b;

    return s.substr(a, b - a);
}

static bool parse_bool(const std::string& s) {
    std::string v;
    v.reserve(s.size());

    for (char c : s) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            v.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(c))
            ));
        }
    }

    return v == "true"
        || v == "1"
        || v == "yes"
        || v == "on";
}

static bool is_none_book(const std::string& path) {
    return path.empty()
        || path == "None"
        || path == "none"
        || path == "NONE"
        || path == "<none>"
        || path == "null"
        || path == "-";
}

static std::string epd_to_fen(const std::string& line_raw) {
    std::string line = trim_ws(line_raw);

    if (line.empty() || line[0] == '#' || line[0] == ';')
        return "";

    auto comment = line.find(';');
    if (comment != std::string::npos) {
        line = trim_ws(line.substr(0, comment));
        if (line.empty()) return "";
    }

    auto tokens = split(line, ' ');

    if (tokens.size() < 4) return "";
    if (tokens[0].find('/') == std::string::npos) return "";

    std::string fen = tokens[0] + " " + tokens[1] + " " + tokens[2] + " " + tokens[3];

    if (tokens.size() >= 6 && is_integer(tokens[4]) && is_integer(tokens[5])) {
        fen += " " + tokens[4] + " " + tokens[5];
    } else {
        fen += " 0 1";
    }

    return fen;
}

static std::vector<std::string> load_epd_book(const std::string& path) {
    std::vector<std::string> fens;

    std::ifstream in(path);
    if (!in) return fens;

    std::string line;
    while (std::getline(in, line)) {
        std::string fen = epd_to_fen(line);
        if (!fen.empty()) fens.push_back(std::move(fen));
    }

    return fens;
}

static bool play_random_plies(chess::Board& board, int plies, std::mt19937_64& rng) {
    for (int i = 0; i < plies; ++i) {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        if (moves.empty()) return false;

        std::uniform_int_distribution<int> dist(0, static_cast<int>(moves.size()) - 1);
        board.makeMove(moves[dist(rng)]);
    }

    return true;
}

static void run_genfen(chess::Board& board, ThreadInfo& thread,
                       int count, uint64_t seed, const std::string& book_path,
                       int random_plies, int eval_limit) {
    if (count <= 0) return;

    if (random_plies < 0) random_plies = 0;

    std::vector<std::string> book;
    bool use_book = !is_none_book(book_path);

    if (use_book) {
        book = load_epd_book(book_path);

        if (book.empty()) {
            std::cout << "info string genfen failed to load book: " << book_path
                      << " ; using startpos" << std::endl;
            use_book = false;
        } else {
            std::cout << "info string genfen loaded " << book.size()
                      << " book positions from " << book_path << std::endl;
        }
    }

    std::mt19937_64 rng(seed);

    int generated = 0;
    int attempts = 0;
    const int max_attempts = std::max(count * 200, 1000);

    while (generated < count && attempts < max_attempts) {
        ++attempts;

        if (use_book) {
            std::uniform_int_distribution<size_t> book_dist(0, book.size() - 1);
            board.setFen(book[book_dist(rng)]);
        } else {
            board.setFen(chess::constants::STARTPOS);
        }

        if (!play_random_plies(board, random_plies, rng))
            continue;

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);

        if (legal.empty())
            continue;

        thread.accumulatorStack.resetAccumulators(board);

        int val = g_nnue.evaluate(board, thread);

        if (val < -eval_limit || val > eval_limit)
            continue;

        std::cout << "info string genfens " << board.getFen() << std::endl;
        std::cout.flush();

        ++generated;
    }

    if (generated < count) {
        std::cout << "info string genfen only produced " << generated
                  << " / " << count << " positions" << std::endl;
        std::cout.flush();
    }

    board.setFen(chess::constants::STARTPOS);
    thread.accumulatorStack.resetAccumulators(board);
}

static bool process_command(const std::string& line, chess::Board& board, ThreadInfo& thread) {
    auto tokens = split(line, ' ');

    if (tokens.empty())
        return true;

    std::string command = tokens[0];

    if (command == "uci") {
        std::cout << "id name Kociolek-2.2" << std::endl;
        std::cout << "id author Kociolek" << std::endl;
        std::cout << "option name Hash type spin default 256 min 1 max 1024" << std::endl;
        std::cout << "option name Threads type spin default 1 min 1 max 256" << std::endl;
        std::cout << "option name EvalFile type string default " << EVALFILE << std::endl;
        std::cout << "option name PolicyFile type string default " << POLICYFILE << std::endl;
        std::cout << "option name PolicyFileSmall type string default " << POLICYFILE_SMALL << std::endl;
        std::cout << "option name UsePolicy type check default true" << std::endl;
        std::cout << "uciok" << std::endl;
        std::cout.flush();
    } else if (command == "setoption") {
        if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "Hash" && tokens[3] == "value") {
            int mb = std::stoi(tokens[4]);
            initTT(mb);
        } else if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "Threads" && tokens[3] == "value") {
            g_num_threads = std::clamp(std::stoi(tokens[4]), 1, 256);
        } else if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "EvalFile" && tokens[3] == "value") {
            std::string path;

            for (size_t i = 4; i < tokens.size(); ++i) {
                path += tokens[i];
                if (i + 1 < tokens.size()) path += " ";
            }

            g_nnue.loadNetwork(path);

            board.setFen(chess::constants::STARTPOS);
            thread.accumulatorStack.resetAccumulators(board);
        } else if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "PolicyFile" && tokens[3] == "value") {
            std::string path;

            for (size_t i = 4; i < tokens.size(); ++i) {
                path += tokens[i];
                if (i + 1 < tokens.size()) path += " ";
            }

            if (!g_policy.load(path)) {
                std::cout << "info string PolicyFile load failed; keeping previous net" << std::endl;
            }
        } else if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "PolicyFileSmall" && tokens[3] == "value") {
            std::string path;

            for (size_t i = 4; i < tokens.size(); ++i) {
                path += tokens[i];
                if (i + 1 < tokens.size()) path += " ";
            }

            if (!g_policy_small.load(path)) {
                std::cout << "info string PolicyFileSmall load failed; keeping previous small net" << std::endl;
            }
        } else if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "UsePolicy" && tokens[3] == "value") {
            g_use_policy = parse_bool(tokens[4]);
        }
    } else if (command == "bench") {
        run_bench(thread);
    } else if (command == "policyoff") {
        g_use_policy = false;
        std::cout << "info string UsePolicy false" << std::endl;
        std::cout.flush();
    } else if (command == "policyon") {
        g_use_policy = true;
        std::cout << "info string UsePolicy true" << std::endl;
        std::cout.flush();
    } else if (command == "genfens" || command == "genfen") {
        int count = 1;
        uint64_t seed = 0;
        std::string book_path;
        int random_plies = GENFEN_RANDOM_PLIES;
        int eval_limit = GENFEN_EVAL_LIMIT;

        for (size_t i = 1; i < tokens.size(); ++i) {
            const std::string& t = tokens[i];

            if (t == "seed" && i + 1 < tokens.size()) {
                try { seed = std::stoull(tokens[++i]); } catch (...) {}
            } else if (t == "book" && i + 1 < tokens.size()) {
                book_path = tokens[++i];
            } else if ((t == "randomply" || t == "randomplies" || t == "plies")
                       && i + 1 < tokens.size()) {
                try { random_plies = std::stoi(tokens[++i]); } catch (...) {}
            } else if ((t == "evallimit" || t == "eval")
                       && i + 1 < tokens.size()) {
                try { eval_limit = std::stoi(tokens[++i]); } catch (...) {}
            } else if (is_integer(t) && i == 1) {
                try { count = std::stoi(t); } catch (...) {}
            }
        }

        run_genfen(board, thread, count, seed, book_path, random_plies, eval_limit);
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
        g_correctionHistory.clear();
        g_pawnCorrectionHistory.clear();
        g_materialCorrectionHistory.clear();
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

        tm.init(time_left, inc, movestogo, movetime,
                board.fullMoveNumber() * 2 - (board.sideToMove() == chess::Color::WHITE ? 2 : 1));

        chess::Move best = search(board, depth, thread, tm, nodes);

        std::cout << "bestmove " << chess::uci::moveToUci(best) << std::endl;
        std::cout.flush();
    } else if (command == "quit") {
        return false;
    } else if (command == "eval") {
        int val = g_nnue.evaluate(board, thread);
        std::cout << "NNUE static eval: " << val << std::endl;
    } else if (command == "policy" || command == "policydebug") {
        int topN = 16;

        if (tokens.size() >= 2) {
            try {
                topN = std::stoi(tokens[1]);
            } catch (...) {
            }
        }

        g_policy.debugPosition(board, topN);
    } else if (command == "policysmall" || command == "policysmalldebug") {
        int topN = 16;

        if (tokens.size() >= 2) {
            try {
                topN = std::stoi(tokens[1]);
            } catch (...) {
            }
        }

        g_policy_small.debugPosition(board, topN);
    } else if (command == "policyhit") {
        int depth = 10;
        int64_t nodes = 0;

        for (size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i] == "depth" && i + 1 < tokens.size()) {
                depth = std::stoi(tokens[i + 1]);
                nodes = 0;
                ++i;
            } else if (tokens[i] == "nodes" && i + 1 < tokens.size()) {
                nodes = std::stoll(tokens[i + 1]);
                depth = 0;
                ++i;
            } else if (is_integer(tokens[i])) {
                depth = std::stoi(tokens[i]);
                nodes = 0;
            }
        }

        runPolicyHitBench(depth, nodes, thread);
        thread.accumulatorStack.resetAccumulators(board);
    } else if (command == "policymove") {
        if (tokens.size() < 2) {
            std::cout << "info string usage: policymove <uci>" << std::endl;
        } else {
            chess::Move m = chess::uci::uciToMove(board, tokens[1]);

            if (m == chess::Move()) {
                std::cout << "info string invalid move " << tokens[1] << std::endl;
            } else {
                g_policy.debugMove(board, m);
            }
        }
    } else if (command == "policysmallmove") {
        if (tokens.size() < 2) {
            std::cout << "info string usage: policysmallmove <uci>" << std::endl;
        } else {
            chess::Move m = chess::uci::uciToMove(board, tokens[1]);

            if (m == chess::Move()) {
                std::cout << "info string invalid move " << tokens[1] << std::endl;
            } else {
                g_policy_small.debugMove(board, m);
            }
        }
    } else if (command == "debug") {
        thread.accumulatorStack.resetAccumulators(board);
        g_nnue.debugNetwork(board, thread.accumulatorStack.current());
    } else if (command == "buckets") {
        thread.accumulatorStack.resetAccumulators(board);
        g_nnue.showBuckets(&board, thread.accumulatorStack.current());
    } else if (command == "d" || command == "display") {
        std::cout << board << std::endl;
        std::cout << "FEN: " << board.getFen() << std::endl;
        std::cout << "Side to move: "
                  << (board.sideToMove() == chess::Color::WHITE ? "White" : "Black")
                  << std::endl;
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
    g_nnue.loadNetwork(EVALFILE);
    std::cout << "info string NNUE loaded" << std::endl;

#if HAVE_POLICY_EMBED
    std::cout << "info string Loading Policy (embedded)..." << std::endl;

    if (!g_policy.loadFromMemory(g_policy_embed_data, g_policy_embed_size, "embedded")) {
        std::cout << "info string Policy embedded load FAILED; trying " << POLICYFILE << std::endl;

        if (!g_policy.load(POLICYFILE)) {
            std::cout << "info string Policy not loaded" << std::endl;
        }
    }
#else
    std::cout << "info string Loading Policy from " << POLICYFILE << "..." << std::endl;

    if (!g_policy.load(POLICYFILE)) {
        std::cout << "info string Policy not loaded" << std::endl;
    }
#endif

#if HAVE_POLICY_SMALL_EMBED
    std::cout << "info string Loading Small Policy (embedded)..." << std::endl;

    if (!g_policy_small.loadFromMemory(g_policy_small_embed_data,
                                       g_policy_small_embed_size,
                                       "embedded_small")) {
        std::cout << "info string Small policy embedded load FAILED; trying "
                  << POLICYFILE_SMALL << std::endl;

        if (!g_policy_small.load(POLICYFILE_SMALL)) {
            std::cout << "info string Small policy not loaded" << std::endl;
        }
    }
#else
    std::cout << "info string Loading Small Policy from " << POLICYFILE_SMALL << "..." << std::endl;

    if (!g_policy_small.load(POLICYFILE_SMALL)) {
        std::cout << "info string Small policy not loaded" << std::endl;
    }
#endif

    board.setFen(chess::constants::STARTPOS);
    thread.accumulatorStack.resetAccumulators(board);

    if (argc > 1) {
        std::string first = argv[1];

        if (first == "genfens" || first == "genfen") {
            std::string joined;

            for (int i = 1; i < argc; ++i) {
                if (i > 1) joined += " ";
                joined += argv[i];
            }

            process_command(joined, board, thread);
        } else {
            for (int i = 1; i < argc; ++i) {
                if (!process_command(argv[i], board, thread))
                    break;
            }
        }

        delete[] tt;
        return;
    }

    std::string line;

    while (std::getline(std::cin, line)) {
        if (!process_command(line, board, thread))
            break;
    }

    delete[] tt;
}