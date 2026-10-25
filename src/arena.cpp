// ============================================================================
// arena.cpp  --  NNUE vs Policy match runner
// ----------------------------------------------------------------------------
// A small self-contained "play them against each other" application:
//
//   * loads the NNUE network (default: EVALFILE, e.g. quantised-v14.bin)
//   * loads the policy network (default: POLICYFILE, e.g. Policy-big.bin)
//   * plays the searching NNUE player against the raw policy player
//   * has an on/off switch for the quiescence search (--qs on|off|both)
//
// Build:   make arena
// Usage:   ./arena --games 20 --qs both --depth 8
//          ./arena --help
// ============================================================================

#include "chess.hpp"
#include "nnue.h"
#include "policy.h"
#include "search.h"
#include "tt.h"
#include "history.h"
#include "zobrist.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef EVALFILE
#define EVALFILE "quantised-v14.bin"
#endif

#ifndef POLICYFILE
#define POLICYFILE "Policy-big.bin"
#endif

// ============================================================================
// Options
// ============================================================================

enum class QsMode { Off, On, Both };

struct Options {
    int  games              = 20;

    std::string nnuePath    = EVALFILE;
    std::string policyPath  = POLICYFILE;
    bool useSmallPolicy     = false;

    int      depth          = 8;      // NNUE search depth per move
    int64_t  nodes          = 0;      // NNUE node limit per move (0 = off)
    int      movetime       = 0;      // NNUE movetime per move in ms (0 = off)

    QsMode   qs             = QsMode::On;
    bool     searchPolicy   = true;   // let the NNUE player use policy ordering

    float    policyTemp     = 0.0f;   // 0 = argmax, >0 = sample from policy^(1/T)
    int      maxPlies       = 240;    // adjudicate as a draw after this many plies
    int      adjudicateCp   = 0;      // >0: adjudicate on |NNUE eval| at maxPlies

    int      openPlies      = 6;      // random opening plies from startpos
    std::string openingsFile;         // or: one FEN per line

    uint64_t seed           = 20240215;
    int      hashMb         = 128;

    std::string pgnPath;
    bool     verbose        = false;
};

// ============================================================================
// Small helpers
// ============================================================================

static std::string nowDateString() {
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d.%02d.%02d",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday);
    return std::string(buf);
}

static double nowMs() {
    using clock = std::chrono::high_resolution_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

static void clearSearchState() {
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
}

static std::string colorName(chess::Color c) {
    return c == chess::Color::WHITE ? "white" : "black";
}

// Startpos FEN -> full move number / side to move, used for PGN numbering.
static void parseFenCounters(const std::string& fen, int& fullmove, bool& whiteToMove) {
    fullmove = 1;
    whiteToMove = true;

    std::istringstream iss(fen);
    std::vector<std::string> parts;
    std::string tok;

    while (iss >> tok) parts.push_back(tok);

    if (parts.size() >= 2) whiteToMove = (parts[1] != "b");
    if (parts.size() >= 6) {
        try {
            fullmove = std::max(1, std::stoi(parts[5]));
        } catch (...) {
            fullmove = 1;
        }
    }
}

// ============================================================================
// Results
// ============================================================================

enum class Outcome { Draw, WhiteWin, BlackWin };

struct GameReport {
    Outcome outcome       = Outcome::Draw;
    std::string reason;

    bool nnueIsWhite      = true;
    int  plies            = 0;

    std::string startFen;                 // empty == startpos
    std::vector<std::string> moves;       // uci, including opening moves

    int      lastNnueScore = 0;           // last NNUE search score, white-relative
    uint64_t nnueNodes     = 0;
    double   nnueMs        = 0.0;
    double   policyMs      = 0.0;
    int      policyMoves   = 0;
};

struct MatchResult {
    bool qs                = true;
    int  games             = 0;
    int  nnueWins          = 0;
    int  policyWins        = 0;
    int  draws             = 0;
    uint64_t nnueNodes     = 0;
    double nnueMs          = 0.0;
    double policyMs        = 0.0;
    std::vector<GameReport> reports;

    double nnueScore() const {
        if (games <= 0) return 0.0;
        return (nnueWins + 0.5 * draws) / double(games);
    }
};

static std::string outcomeToResultString(Outcome o) {
    switch (o) {
        case Outcome::WhiteWin: return "1-0";
        case Outcome::BlackWin: return "0-1";
        default:                return "1/2-1/2";
    }
}

static std::string pgnResultFor(const GameReport& g) {
    return outcomeToResultString(g.outcome);
}

// Score (0..1) -> Elo diff, plus a rough 95% interval.
static void eloFromScore(double score, int games, double& elo, double& lo, double& hi) {
    auto conv = [](double s) {
        if (s <= 0.0) return -1000.0;
        if (s >= 1.0) return 1000.0;
        return -400.0 * std::log10(1.0 / s - 1.0);
    };

    elo = conv(score);

    if (games <= 0) {
        lo = hi = elo;
    } else {
        const double se = std::sqrt(std::max(0.0, score * (1.0 - score)) / double(games));
        lo = conv(std::clamp(score - 1.96 * se, 0.0, 1.0));
        hi = conv(std::clamp(score + 1.96 * se, 0.0, 1.0));
    }

    // Avoid printing "-0".
    if (std::abs(elo) < 1.0) elo = 0.0;
    if (std::abs(lo)  < 1.0) lo  = 0.0;
    if (std::abs(hi)  < 1.0) hi  = 0.0;
}

// Per-process RNG used by the policy sampler (defined below).
static std::mt19937_64& threadRng();

// ============================================================================
// Players
// ============================================================================

// Raw policy player: one forward pass, no search at all.
static chess::Move policyBestMove(const chess::Board& board,
                                  const chess::Movelist& legal,
                                  const Options& opt,
                                  std::mt19937_64& rng,
                                  bool& usedPolicy) {
    usedPolicy = false;

    RootPolicy rp;
    const bool ok = opt.useSmallPolicy ? computeSmallRootPolicy(board, rp)
                                       : computeRootPolicy(board, rp);

    if (ok && rp.nlegal > 0) {
        usedPolicy = true;

        if (opt.policyTemp <= 0.0f) {
            int best = 0;
            for (int i = 1; i < rp.nlegal; ++i) {
                if (rp.prob_any[i] > rp.prob_any[best]) best = i;
            }
            return rp.legals[best];
        }

        std::vector<double> w(static_cast<size_t>(rp.nlegal), 0.0);
        double sum = 0.0;

        for (int i = 0; i < rp.nlegal; ++i) {
            const double p = std::max(1e-12, double(rp.prob_any[i]));
            w[static_cast<size_t>(i)] = std::pow(p, 1.0 / double(opt.policyTemp));
            sum += w[static_cast<size_t>(i)];
        }

        if (sum > 0.0) {
            std::uniform_real_distribution<double> dist(0.0, sum);
            double x = dist(rng);

            for (int i = 0; i < rp.nlegal; ++i) {
                x -= w[static_cast<size_t>(i)];
                if (x <= 0.0) return rp.legals[i];
            }

            return rp.legals[rp.nlegal - 1];
        }
    }

    // Fallback: uniform random legal move.
    std::uniform_int_distribution<int> dist(0, int(legal.size()) - 1);
    return legal[dist(rng)];
}

// ============================================================================
// Game
// ============================================================================

static std::mt19937_64& threadRng() {
    static std::mt19937_64 rng(0xC0FFEE1234ULL);
    return rng;
}

static bool isLegalMove(const chess::Movelist& legal, const chess::Move& m) {
    for (int i = 0; i < int(legal.size()); ++i) {
        if (legal[i] == m) return true;
    }
    return false;
}

static void applyMove(chess::Board& board, ThreadInfo& thread, const chess::Move& m) {
    board.makeMove(m);
    thread.accumulatorStack.resetAccumulators(board);
}

static GameReport playGame(const Options& opt,
                           bool nnueIsWhite,
                           bool useQs,
                           int openingIndex,
                           const std::vector<std::string>& openingFens,
                           int gameNumber) {
    GameReport rep;
    rep.nnueIsWhite = nnueIsWhite;

    chess::Board board;
    board.set960(false);

    ThreadInfo thread;

    const bool fromFenFile = !openingFens.empty();

    if (fromFenFile) {
        const std::string& fen = openingFens[static_cast<size_t>(openingIndex) % openingFens.size()];
        board.setFen(fen);
        rep.startFen = fen;
    } else {
        board.setFen(chess::constants::STARTPOS);
    }

    thread.accumulatorStack.resetAccumulators(board);

    clearSearchState();
    g_use_qs     = useQs;
    g_use_policy = opt.searchPolicy;
    g_silent     = true;

    // ---- Opening book / random opening ------------------------------------
    if (!fromFenFile && opt.openPlies > 0) {
        std::mt19937_64 orng(opt.seed * 1000003ULL + uint64_t(openingIndex) * 7919ULL + 17ULL);

        for (int i = 0; i < opt.openPlies; ++i) {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board);

            if (legal.empty()) break;
            if (board.isRepetition() || board.isHalfMoveDraw()) break;

            std::uniform_int_distribution<int> dist(0, int(legal.size()) - 1);
            const chess::Move m = legal[dist(orng)];

            rep.moves.push_back(chess::uci::moveToUci(m));

            if (opt.verbose) {
                std::cout << "  game " << gameNumber
                          << " ply " << std::right << std::setw(3) << (i + 1)
                          << "  " << std::left << std::setw(6) << "open"
                          << " (" << colorName(board.sideToMove()) << ") "
                          << chess::uci::moveToUci(m)
                          << std::endl;
            }

            applyMove(board, thread, m);
        }
    }

    // ---- Main move loop ---------------------------------------------------
    for (int ply = 0; ply < opt.maxPlies; ++ply) {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);

        if (legal.empty()) {
            if (board.inCheck()) {
                // Side to move is checkmated.
                const bool whiteMated = (board.sideToMove() == chess::Color::WHITE);
                rep.outcome = whiteMated ? Outcome::BlackWin : Outcome::WhiteWin;
                rep.reason  = "checkmate";
            } else {
                rep.outcome = Outcome::Draw;
                rep.reason  = "stalemate";
            }
            break;
        }

        if (board.isHalfMoveDraw()) {
            rep.outcome = Outcome::Draw;
            rep.reason  = "fifty-move rule";
            break;
        }

        if (board.isRepetition()) {
            rep.outcome = Outcome::Draw;
            rep.reason  = "threefold repetition";
            break;
        }

        if (board.isInsufficientMaterial()) {
            rep.outcome = Outcome::Draw;
            rep.reason  = "insufficient material";
            break;
        }

        const bool whiteToMove = (board.sideToMove() == chess::Color::WHITE);
        const bool nnueTurn    = (whiteToMove == nnueIsWhite);

        chess::Move move;
        std::string moveInfo;

        if (nnueTurn) {
            const double t0 = nowMs();

            // TimeManager shaves MOVE_OVERHEAD_MS off the movetime budget, so
            // budgets that small would stop the search before move 1.
            int movetime = opt.movetime;
            if (movetime > 0 && movetime <= TimeManager::MOVE_OVERHEAD_MS)
                movetime = TimeManager::MOVE_OVERHEAD_MS + 25;

            TimeManager tm;
            tm.init(movetime, 0, 0, movetime, ply);

            int score = 0;
            uint64_t nodes = 0;

            move = search(board, opt.depth, thread, tm, opt.nodes, &score, &nodes);

            if (move == chess::Move() || !isLegalMove(legal, move))
                move = legal[0];

            const double dt = nowMs() - t0;

            rep.nnueMs    += dt;
            rep.nnueNodes += nodes;
            // search() returns a side-to-move relative score; store it white-relative
            rep.lastNnueScore = nnueIsWhite ? score : -score;

            if (opt.verbose) {
                std::ostringstream oss;
                oss << " d" << opt.depth;
                if (opt.nodes > 0) oss << " n" << opt.nodes;
                if (opt.movetime > 0) oss << " t" << opt.movetime;
                // score is side-to-move relative, as returned by search()
                oss << " cp " << score << " nodes " << nodes;
                moveInfo = oss.str();
            }
        } else {
            const double t0 = nowMs();

            bool usedPolicy = false;
            move = policyBestMove(board, legal, opt, threadRng(), usedPolicy);

            if (!isLegalMove(legal, move)) move = legal[0];

            const double dt = nowMs() - t0;

            rep.policyMs += dt;
            rep.policyMoves++;

            if (opt.verbose) {
                moveInfo = usedPolicy ? " (policy)" : " (policy unavailable -> random)";
            }
        }

        rep.moves.push_back(chess::uci::moveToUci(move));
        rep.plies = ply + 1;

        if (opt.verbose) {
            std::cout << "  game " << gameNumber
                      << " ply " << std::right << std::setw(3) << (ply + 1)
                      << "  " << std::left << std::setw(6) << (nnueTurn ? "NNUE" : "Policy")
                      << " (" << colorName(board.sideToMove()) << ") "
                      << chess::uci::moveToUci(move)
                      << moveInfo
                      << std::endl;
            std::cout.flush();
        }

        applyMove(board, thread, move);
    }

    // ---- Adjudication at the ply cap --------------------------------------
    if (rep.reason.empty()) {
        rep.plies = int(rep.moves.size());

        if (opt.adjudicateCp > 0) {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board);

            if (legal.empty()) {
                if (board.inCheck()) {
                    const bool whiteMated = (board.sideToMove() == chess::Color::WHITE);
                    rep.outcome = whiteMated ? Outcome::BlackWin : Outcome::WhiteWin;
                    rep.reason = "checkmate";
                } else {
                    rep.outcome = Outcome::Draw;
                    rep.reason = "stalemate";
                }
            } else {
                const int eval = g_nnue.evaluate(board, thread);

                if (std::abs(eval) >= opt.adjudicateCp) {
                    const bool stmWins = (eval > 0);
                    const bool whiteWins =
                        (board.sideToMove() == chess::Color::WHITE) == stmWins;

                    rep.outcome = whiteWins ? Outcome::WhiteWin : Outcome::BlackWin;
                    rep.reason  = "adjudicated (eval " + std::to_string(eval) + ")";
                } else {
                    rep.outcome = Outcome::Draw;
                    rep.reason  = "adjudicated draw (ply cap)";
                }
            }
        } else {
            rep.outcome = Outcome::Draw;
            rep.reason  = "adjudicated draw (ply cap)";
        }
    }

    return rep;
}

// ============================================================================
// Match
// ============================================================================

static MatchResult runMatch(const Options& opt, bool useQs) {
    MatchResult res;
    res.qs = useQs;

    std::vector<std::string> openingFens;

    if (!opt.openingsFile.empty()) {
        std::ifstream in(opt.openingsFile);

        if (!in) {
            std::cerr << "error: cannot open openings file: " << opt.openingsFile << std::endl;
        } else {
            std::string line;

            while (std::getline(in, line)) {
                // trim
                size_t a = line.find_first_not_of(" \t\r\n");
                if (a == std::string::npos) continue;
                size_t b = line.find_last_not_of(" \t\r\n");
                std::string f = line.substr(a, b - a + 1);

                if (f.empty() || f[0] == '#') continue;
                openingFens.push_back(f);
            }
        }

        if (openingFens.empty()) {
            std::cout << "info: no usable FENs in " << opt.openingsFile
                      << ", falling back to random openings" << std::endl;
        }
    }

    std::cout << "=====================================================================\n";
    std::cout << " match: Kociolek NNUE (search)  vs  policy net (no search)\n";
    std::cout << " quiescence search : " << (useQs ? "ON" : "OFF") << "\n";
    std::cout << " nnue player       : depth " << opt.depth;
    if (opt.nodes > 0)    std::cout << ", nodes " << opt.nodes;
    if (opt.movetime > 0) std::cout << ", movetime " << opt.movetime << "ms";
    std::cout << ", search-policy " << (opt.searchPolicy ? "on" : "off") << "\n";
    std::cout << " policy player     : " << (opt.useSmallPolicy ? "small" : "big")
              << " net, " << (opt.policyTemp <= 0.0f
                                ? std::string("argmax")
                                : ("sample T=" + std::to_string(opt.policyTemp)))
              << "\n";
    std::cout << " games             : " << opt.games << "\n";
    std::cout << "=====================================================================\n";
    std::cout.flush();

    for (int g = 0; g < opt.games; ++g) {
        // Alternate colours; both colours of the same opening pair use the
        // same opening (opening index = g / 2).
        const bool nnueIsWhite = (g % 2 == 0);
        const int  openingIdx  = g / 2;

        const double t0 = nowMs();
        GameReport rep = playGame(opt, nnueIsWhite, useQs, openingIdx, openingFens, g + 1);
        const double dt = nowMs() - t0;

        res.games++;
        res.nnueNodes += rep.nnueNodes;
        res.nnueMs    += rep.nnueMs;
        res.policyMs  += rep.policyMs;

        const bool nnueWon =
            (rep.outcome == Outcome::WhiteWin && nnueIsWhite) ||
            (rep.outcome == Outcome::BlackWin && !nnueIsWhite);

        const bool policyWon =
            (rep.outcome == Outcome::WhiteWin && !nnueIsWhite) ||
            (rep.outcome == Outcome::BlackWin && nnueIsWhite);

        if (nnueWon)        res.nnueWins++;
        else if (policyWon) res.policyWins++;
        else                res.draws++;

        res.reports.push_back(rep);

        const std::string resultStr = outcomeToResultString(rep.outcome);

        std::cout << "game " << std::right << std::setw(3) << (g + 1) << "/" << opt.games
                  << "  " << (nnueIsWhite ? "NNUE=w Policy=b" : "NNUE=b Policy=w")
                  << "  " << resultStr
                  << "  (" << rep.reason << ")"
                  << "  plies " << rep.plies
                  << "  " << std::fixed << std::setprecision(1) << (dt / 1000.0) << "s"
                  << "  nnue_nodes " << rep.nnueNodes
                  << std::endl;
        std::cout.flush();
    }

    std::cout << "---------------------------------------------------------------------\n";
    std::cout << " NNUE wins " << res.nnueWins
              << "   policy wins " << res.policyWins
              << "   draws " << res.draws
              << "   (" << res.games << " games)\n";

    const double score = res.nnueScore();
    double elo = 0.0, lo = 0.0, hi = 0.0;
    eloFromScore(score, res.games, elo, lo, hi);

    std::cout << " NNUE score " << std::fixed << std::setprecision(2)
              << (score * 100.0) << "%   Elo "
              << std::showpos << std::setprecision(0) << elo << std::noshowpos
              << "  [" << lo << ", " << hi << "]" << std::endl;

    if (res.nnueMs > 0.0) {
        std::cout << " nnue time " << std::setprecision(2) << (res.nnueMs / 1000.0) << "s"
                  << "   policy time " << (res.policyMs / 1000.0) << "s" << std::endl;
    }

    std::cout << "=====================================================================\n";
    std::cout.flush();

    return res;
}

// ============================================================================
// PGN output
// ============================================================================

static std::string pgnMoves(const GameReport& g) {
    std::ostringstream out;

    int fullmove = 1;
    bool whiteToMove = true;

    if (!g.startFen.empty()) {
        parseFenCounters(g.startFen, fullmove, whiteToMove);
    }

    bool first = true;

    for (size_t i = 0; i < g.moves.size(); ++i) {
        if (whiteToMove) {
            out << (first ? "" : " ") << fullmove << ". " << g.moves[i];
        } else {
            if (first) out << fullmove << "... ";
            else       out << " ";
            out << g.moves[i];
            ++fullmove;
        }

        first = false;
        whiteToMove = !whiteToMove;
    }

    if (g.moves.empty()) out << "";

    return out.str();
}

static void appendPgn(std::ostream& out,
                      const Options& opt,
                      const MatchResult& match,
                      int matchIndex) {
    const std::string date = nowDateString();

    int round = 0;

    for (const GameReport& g : match.reports) {
        ++round;

        const std::string nnueName =
            "Kociolek-NNUE d" + std::to_string(opt.depth) +
            (match.qs ? " qs=on" : " qs=off");

        const std::string policyName =
            std::string("Policy-") + (opt.useSmallPolicy ? "small" : "big") +
            (opt.policyTemp <= 0.0f
                 ? " argmax"
                 : (" T=" + std::to_string(opt.policyTemp)));

        out << "[Event \"NNUE vs Policy arena\"]\n";
        out << "[Site \"Kociolek-2.3 arena\"]\n";
        out << "[Date \"" << date << "\"]\n";
        out << "[Round \"" << (matchIndex * 1000 + round) << "\"]\n";
        out << "[White \"" << (g.nnueIsWhite ? nnueName : policyName) << "\"]\n";
        out << "[Black \"" << (g.nnueIsWhite ? policyName : nnueName) << "\"]\n";
        out << "[Result \"" << pgnResultFor(g) << "\"]\n";
        out << "[Termination \"" << g.reason << "\"]\n";
        out << "[PlyCount \"" << g.plies << "\"]\n";

        if (!g.startFen.empty()) {
            out << "[SetUp \"1\"]\n";
            out << "[FEN \"" << g.startFen << "\"]\n";
        }

        out << "\n" << pgnMoves(g) << " " << pgnResultFor(g) << "\n\n";
    }
}

// ============================================================================
// CLI
// ============================================================================

static void printUsage() {
    std::cout <<
R"(arena - Kociolek NNUE vs policy net match runner

usage: ./arena [options]

Match
  --games N            games to play, colours alternate          (default 20)
  --qs on|off|both     quiescence search switch; "both" plays the
                       whole match twice (qs on, then qs off)    (default on)
  --depth N            NNUE search depth per move                 (default 8)
  --nodes N            NNUE node limit per move (0 = off)         (default 0)
  --movetime MS        NNUE time per move in ms, min 75 (0 = off)  (default 0)
  --search-policy on|off
                       let the NNUE player use the policy net for
                       move ordering / time management            (default on)

Networks
  --nnue <file>        NNUE network file                          (default )" EVALFILE R"()
  --policy <file>      policy network file                        (default )" POLICYFILE R"()
  --policy-small       use the small policy net (POLICY_SMALL_HL)
  --policy-temp T      policy player: 0 = argmax, >0 = sample
                       from policy^(1/T)                          (default 0)

Game handling
  --open-plies N       random opening plies from startpos         (default 6)
  --openings <file>    one FEN per line instead of random openings
  --max-plies N        adjudicate a draw after N plies            (default 240)
  --adjudicate-cp N    >0: adjudicate on |NNUE eval| at max-plies (default 0)
  --hash MB            transposition table size                   (default 128)
  --seed N             random seed, controls openings             (default 20240215)

Output
  --pgn <file>         write the games as PGN
  --verbose            print every move
  --help               this text

examples
  ./arena                                    # 20 games, qs on
  ./arena --games 40 --qs both --depth 8     # qs on vs qs off
  ./arena --qs off --movetime 100 --verbose
  ./arena --openings openings.fen --policy-temp 0.5 --pgn match.pgn
)";
}

static bool parseOnOff(const std::string& s, bool& out) {
    std::string v;
    for (char c : s) v.push_back(char(std::tolower(static_cast<unsigned char>(c))));

    if (v == "on" || v == "true" || v == "1" || v == "yes")  { out = true;  return true; }
    if (v == "off" || v == "false" || v == "0" || v == "no") { out = false; return true; }
    return false;
}

static bool parseArgs(int argc, char** argv, Options& opt, bool& showHelp) {
    showHelp = false;

    auto needValue = [&](int& i, const std::string& flag) -> const char* {
        if (i + 1 >= argc) {
            std::cerr << "error: " << flag << " needs a value" << std::endl;
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        if (a == "--help" || a == "-h") {
            showHelp = true;
            return true;
        } else if (a == "--verbose" || a == "-v") {
            opt.verbose = true;
        } else if (a == "--policy-small") {
            opt.useSmallPolicy = true;
        } else if (a == "--games") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.games = std::max(1, std::atoi(v));
        } else if (a == "--depth") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.depth = std::max(1, std::atoi(v));
        } else if (a == "--nodes") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.nodes = std::max<int64_t>(0, std::atoll(v));
        } else if (a == "--movetime") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.movetime = std::max(0, std::atoi(v));
        } else if (a == "--qs") {
            const char* v = needValue(i, a); if (!v) return false;

            std::string s(v);
            std::transform(s.begin(), s.end(), s.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });

            if (s == "both") {
                opt.qs = QsMode::Both;
            } else {
                bool on = true;
                if (!parseOnOff(s, on)) {
                    std::cerr << "error: --qs expects on|off|both" << std::endl;
                    return false;
                }
                opt.qs = on ? QsMode::On : QsMode::Off;
            }
        } else if (a == "--search-policy") {
            const char* v = needValue(i, a); if (!v) return false;
            bool on = true;
            if (!parseOnOff(v, on)) {
                std::cerr << "error: --search-policy expects on|off" << std::endl;
                return false;
            }
            opt.searchPolicy = on;
        } else if (a == "--nnue") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.nnuePath = v;
        } else if (a == "--policy") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.policyPath = v;
        } else if (a == "--policy-temp") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.policyTemp = float(std::max(0.0, std::atof(v)));
        } else if (a == "--open-plies") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.openPlies = std::max(0, std::atoi(v));
        } else if (a == "--openings") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.openingsFile = v;
        } else if (a == "--max-plies") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.maxPlies = std::max(1, std::atoi(v));
        } else if (a == "--adjudicate-cp") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.adjudicateCp = std::max(0, std::atoi(v));
        } else if (a == "--hash") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.hashMb = std::max(1, std::atoi(v));
        } else if (a == "--seed") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.seed = std::strtoull(v, nullptr, 10);
        } else if (a == "--pgn") {
            const char* v = needValue(i, a); if (!v) return false;
            opt.pgnPath = v;
        } else {
            std::cerr << "error: unknown option '" << a << "' (try --help)" << std::endl;
            return false;
        }
    }

    return true;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
    Options opt;
    bool showHelp = false;

    if (!parseArgs(argc, argv, opt, showHelp))
        return 1;

    if (showHelp) {
        printUsage();
        return 0;
    }

    std::cout << "info string initialising engine...\n";

    initZobrist();
    initLMR();
    initTT(size_t(opt.hashMb));

    g_silent = true;

    std::cout << "info string loading NNUE  " << opt.nnuePath << "\n";
    g_nnue.loadNetwork(opt.nnuePath);

    std::cout << "info string loading policy " << opt.policyPath
              << (opt.useSmallPolicy ? " (as small net)" : "") << "\n";

    bool policyOk = false;
    if (opt.useSmallPolicy) {
        policyOk = g_policy_small.load(opt.policyPath);
        if (!policyOk) {
            std::cerr << "error: failed to load the small policy net from "
                      << opt.policyPath << "\n"
                      << "       (the file must match POLICY_SMALL_HL="
                      << POLICY_SMALL_HL << ")\n";
        }
    } else {
        policyOk = g_policy.load(opt.policyPath);
        if (!policyOk) {
            std::cerr << "error: failed to load the policy net from "
                      << opt.policyPath << "\n"
                      << "       (the file must match KOCIOLEK_POLICY_HL="
                      << KOCIOLEK_POLICY_HL << ")\n";
        }
    }

    if (!policyOk)
        return 1;

    std::cout << "info string policy loaded (hidden layer "
              << (opt.useSmallPolicy ? POLICY_SMALL_HL : KOCIOLEK_POLICY_HL)
              << ", " << (opt.useSmallPolicy ? g_policy_small.num_moves : g_policy.num_moves)
              << " output moves)\n";

    std::vector<MatchResult> matches;

    if (opt.qs == QsMode::Both) {
        matches.push_back(runMatch(opt, true));
        matches.push_back(runMatch(opt, false));
    } else {
        matches.push_back(runMatch(opt, opt.qs == QsMode::On));
    }

    // ---- Comparison table (for --qs both) ---------------------------------
    if (matches.size() > 1) {
        std::cout << "\n";
        std::cout << "================= quiescence search comparison =================\n";
        std::cout << std::left << std::setw(12) << " qs"
                  << std::right << std::setw(7) << "games"
                  << std::setw(7) << "wins"
                  << std::setw(8) << "losses"
                  << std::setw(7) << "draws"
                  << std::setw(9) << "score"
                  << std::setw(8) << "elo"
                  << "\n";

        for (const auto& m : matches) {
            double elo = 0.0, lo = 0.0, hi = 0.0;
            eloFromScore(m.nnueScore(), m.games, elo, lo, hi);

            std::cout << " " << std::left << std::setw(11) << (m.qs ? "on" : "off")
                      << std::right << std::setw(7) << m.games
                      << std::setw(7) << m.nnueWins
                      << std::setw(8) << m.policyWins
                      << std::setw(7) << m.draws
                      << std::setw(8) << std::fixed << std::setprecision(1)
                      << (m.nnueScore() * 100.0) << "%"
                      << std::setw(8) << std::setprecision(0) << std::showpos << elo
                      << std::noshowpos
                      << "\n";
        }

        if (matches.size() == 2) {
            const double dElo = matches[0].nnueScore() - matches[1].nnueScore();
            std::cout << "\n score(qs=on) - score(qs=off) = "
                      << std::showpos << std::fixed << std::setprecision(1)
                      << (dElo * 100.0) << "%" << std::noshowpos
                      << "   (positive means quiescence search helps)\n";
        }

        std::cout << "===============================================================\n";
        std::cout.flush();
    }

    // ---- PGN --------------------------------------------------------------
    if (!opt.pgnPath.empty()) {
        std::ofstream out(opt.pgnPath);

        if (!out) {
            std::cerr << "error: cannot write PGN file " << opt.pgnPath << std::endl;
        } else {
            for (size_t i = 0; i < matches.size(); ++i)
                appendPgn(out, opt, matches[i], int(i) + 1);

            std::cout << "info string wrote PGN to " << opt.pgnPath << std::endl;
        }
    }

    g_use_qs     = true;
    g_use_policy = true;

    return 0;
}
