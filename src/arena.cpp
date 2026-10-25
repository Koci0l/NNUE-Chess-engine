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
#include <filesystem>
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

// How the NNUE side picks its move:
//   Search -> the engine's alpha-beta search (depth/nodes/movetime, QS switch)
//   Static -> no search at all: score every legal move once with the NNUE
//             output and play the best one (exactly d1 / ~#legal-move nodes)
enum class NnueMode { Search, Static };

// One opening position, from a FEN file or an EPD book (e.g. uho.epd).
struct Opening {
    std::string fen;   // normalised FEN (6 fields)
    std::string id;    // EPD "id" op, e.g. UHO_Lichess_4852_v1.epd:137
    std::string bm;    // EPD "bm" op, if any
    bool        epd = false;
};

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
    std::vector<std::string> openingsFiles;  // FEN-per-line / EPD / uho.epd
    std::string openingsOrder = "random";    // random | seq
    bool        epdPlayBm   = false;  // play the EPD "bm" move (only if it is UCI)

    NnueMode nnueMode       = NnueMode::Search;

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
    std::string openingId;                // EPD id, if any
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

// ============================================================================
// Openings: FEN lists and EPD books (uho.epd, ...)
// ----------------------------------------------------------------------------
// Accepted line formats:
//   * FEN      "rnbqkbnr/pppppppp/8/8/... w KQkq - 0 1"
//   * EPD      "rnbqkbnr/... w KQkq - bm e4; id "UHO_Lichess_4852_v1.epd:137";"
//   * UCI      "rnbqkbnr/... w KQkq - moves e2e4 e7e5"   (moves are applied)
// Comment lines ('#') and blank lines are ignored, and a UTF-8 BOM is stripped.
// ============================================================================

static std::string trimCopy(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();

    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> splitWs(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) out.push_back(t);
    return out;
}

// Reads the value of an EPD operation ("op value;"), empty if not present.
static std::string epdOp(const std::string& line, const std::string& op) {
    const std::string key = op + " ";
    size_t pos = line.find(key);

    while (pos != std::string::npos) {
        const bool atStart = (pos == 0) || std::isspace(static_cast<unsigned char>(line[pos - 1])) ||
                             line[pos - 1] == ';';

        if (atStart) {
            size_t start = pos + key.size();

            // Values may be quoted, optionally with a leading '"'.
            if (start < line.size() && line[start] == '"') {
                const size_t end = line.find('"', start + 1);
                if (end == std::string::npos) return line.substr(start + 1);
                return line.substr(start + 1, end - start - 1);
            }

            size_t end = line.find(';', start);
            if (end == std::string::npos) end = line.size();

            return trimCopy(line.substr(start, end - start));
        }

        pos = line.find(key, pos + 1);
    }

    return std::string();
}

// Returns the 4 FEN fields of an EPD/FEN line, or an empty vector.
static std::vector<std::string> boardFields(const std::string& line) {
    std::vector<std::string> toks = splitWs(line);

    if (toks.size() < 4) return {};

    // Drop a trailing ';' from the castling / ep field.
    for (int i = 0; i < 4; ++i) {
        while (!toks[static_cast<size_t>(i)].empty() &&
               toks[static_cast<size_t>(i)].back() == ';') {
            toks[static_cast<size_t>(i)].pop_back();
        }
    }

    if (toks[1] != "w" && toks[1] != "b") return {};

    return { toks[0], toks[1], toks[2], toks[3] };
}

// Builds a full 6-field FEN and plays any "moves ..." from the line.
static bool openingFromLine(const std::string& line,
                            Opening& out,
                            std::string& error) {
    const std::vector<std::string> fields = boardFields(line);

    if (fields.empty()) {
        error = "not a FEN/EPD position";
        return false;
    }

    const std::vector<std::string> toks = splitWs(line);
    const bool isEpd = (line.find(';') != std::string::npos) || !epdOp(line, "id").empty() ||
                       !epdOp(line, "bm").empty();

    int hm = 0;
    int fm = 1;
    size_t next = 4;
    bool hasCounters = false;

    if (toks.size() >= 6 && toks[4].find_first_not_of("0123456789") == std::string::npos &&
        toks[5].find_first_not_of("0123456789") == std::string::npos) {
        hm = std::atoi(toks[4].c_str());
        fm = std::max(1, std::atoi(toks[5].c_str()));
        hasCounters = true;
        next = 6;
    }

    if (isEpd && !hasCounters) {
        const std::string hmvc = epdOp(line, "hmvc");
        const std::string fmvn = epdOp(line, "fmvn");
        if (!hmvc.empty()) hm = std::atoi(hmvc.c_str());
        if (!fmvn.empty()) fm = std::max(1, std::atoi(fmvn.c_str()));
    }

    std::string fen = fields[0] + " " + fields[1] + " " + fields[2] + " " + fields[3] + " " +
                      std::to_string(hm) + " " + std::to_string(fm);

    chess::Board board;

    if (!board.setFen(fen)) {
        error = "illegal FEN: " + fen;
        return false;
    }

    // Optional: "moves e2e4 e7e5" (UCI start position lines).
    if (next < toks.size() && toks[next] == "moves") {
        for (size_t i = next + 1; i < toks.size(); ++i) {
            std::string tok = toks[i];

            while (!tok.empty() && (tok.back() == ';' || tok.back() == '"')) tok.pop_back();

            chess::Move m = chess::uci::uciToMove(board, tok);

            if (m == chess::Move()) break;

            board.makeMove(m);
        }

        fen = board.getFen();
    }

    out = Opening();
    out.fen = fen;
    out.epd = isEpd;
    out.id  = epdOp(line, "id");
    out.bm  = epdOp(line, "bm");

    // Only plain UCI promotions are usable as an opening move.
    if (!out.bm.empty() && out.bm.find('=') != std::string::npos)
        out.bm.clear();

    return true;
}

// "uho.epd" (or whatever --uho was given) may not exist as typed: fall back to
// the usual UHO file names in the working directory, then to any UHO*.epd.
// Returns an empty string when nothing was found.
static std::string resolveBookPath(const std::string& wanted) {
    auto exists = [](const std::string& p) {
        std::ifstream f(p);
        return f.good();
    };

    if (exists(wanted)) return wanted;

    static const char* candidates[] = {
        "uho.epd",
        "UHO_Lichess_4852_v1.epd",
        "UHO_4060_v4.epd",
        "UHO_4060_v3.epd",
        "UHO_4060_v2.epd",
        "UHO_4060_v1.epd",
    };

    for (const char* c : candidates) {
        if (exists(c)) return c;
    }

    // Last resort: any *.epd whose name starts with UHO in the working dir.
#if defined(__cpp_lib_filesystem)
    namespace fs = std::filesystem;

    std::error_code ec;
    std::string best;

    for (const auto& e : fs::directory_iterator(fs::current_path(), ec)) {
        if (!e.is_regular_file(ec)) continue;

        const std::string name = e.path().filename().string();

        if (name.rfind("UHO", 0) == 0 && e.path().extension() == ".epd") {
            if (best.empty() || name < best) best = name;
        }
    }

    return best;
#else
    return std::string();
#endif
}

// Streams the book(s) and keeps only what the match needs, so even a
// multi-million-line book (UHO_Lichess_4852_v1.epd) stays cheap in memory:
//   order == "seq"    -> the first `needed` openings of the file(s)
//   order == "random" -> a uniform reservoir sample of `needed` openings
static bool loadOpenings(const std::vector<std::string>& paths,
                         size_t needed,
                         const std::string& order,
                         uint64_t seed,
                         std::vector<Opening>& out,
                         size_t& totalSeen,
                         std::string& error) {
    out.clear();
    totalSeen = 0;

    std::mt19937_64 rng(seed);

    size_t badLines    = 0;
    size_t badReported = 0;

    for (const std::string& path : paths) {
        std::ifstream in(path);

        if (!in) {
            error = "cannot open openings file: " + path;
            return false;
        }

        std::string line;
        int lineNo = 0;

        while (std::getline(in, line)) {
            ++lineNo;

            if (lineNo == 1 && line.size() >= 3 &&
                static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line.erase(0, 3);
            }

            const std::string t = trimCopy(line);

            if (t.empty() || t[0] == '#') continue;

            Opening op;
            std::string err;

            if (!openingFromLine(t, op, err)) {
                ++badLines;

                if (badReported < 5) {
                    std::cerr << "warning: " << path << ":" << lineNo << ": "
                              << err << " (skipped)" << std::endl;
                    ++badReported;
                }

                continue;
            }

            const size_t n = totalSeen++;

            if (out.size() < needed) {
                out.push_back(std::move(op));
            } else if (order == "seq") {
                // keep the first `needed` only
                continue;
            } else {
                // Reservoir sampling: replace a random slot with probability
                // needed / n, so the kept set is a uniform sample.
                std::uniform_int_distribution<size_t> dist(0, n);
                const size_t j = dist(rng);

                if (j < out.size()) out[j] = std::move(op);
            }
        }
    }

    if (badLines > 0) {
        std::cerr << "warning: skipped " << badLines << " unparsable line(s)" << std::endl;
    }

    if (order == "seq" && out.size() == needed) {
        // already in file order
    } else if (order == "random") {
        std::shuffle(out.begin(), out.end(), rng);
    }

    return true;
}

// Per-process RNG used by the policy sampler (defined below).
static std::mt19937_64& threadRng();

// ============================================================================
// NNUE player without search: score every legal move with the NNUE output
// ----------------------------------------------------------------------------
// One NNUE forward pass per legal move (the leaf eval), i.e. what a depth-1
// search would see but with no quiescence search, no pruning and no ordering.
// On the start position with the policy net this is ~20 nodes -- exactly one
// per legal move.
// ============================================================================

struct StaticNnueResult {
    chess::Move move;
    int         bestMoveScore = 0;   // mover-relative (= NNUE's own point of view)
    int         bestEvalScore = 0;   // white-relative static eval of the position
    uint64_t    nodes         = 0;   // == number of legal moves
    bool        ok            = false;
};

static StaticNnueResult staticNnueBestMove(chess::Board& board, ThreadInfo& thread) {
    StaticNnueResult res;

    chess::Movelist legal;
    chess::movegen::legalmoves(legal, board);

    if (legal.empty()) return res;

    const bool whiteToMove = (board.sideToMove() == chess::Color::WHITE);
    const int  sign        = whiteToMove ? 1 : -1;

    bool haveMoveScore = false;

    for (int i = 0; i < int(legal.size()); ++i) {
        const chess::Move m = legal[static_cast<size_t>(i)];

        board.makeMove(m);
        thread.accumulatorStack.resetAccumulators(board);

        // NNUE is side-to-move relative, so at the child this score is from the
        // *opponent's* point of view; the move is worth -eval to us.
        const int childEval = g_nnue.evaluate(board, thread);
        const int gain      = -childEval;

        board.unmakeMove(m);
        thread.accumulatorStack.resetAccumulators(board);

        res.nodes++;

        if (!haveMoveScore || gain > res.bestMoveScore) {
            res.bestMoveScore = gain;
            res.move = m;
            haveMoveScore = true;
        }
    }

    thread.accumulatorStack.resetAccumulators(board);
    res.bestEvalScore = sign * g_nnue.evaluate(board, thread);
    res.ok = haveMoveScore;

    return res;
}

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
                           const std::vector<Opening>& openings,
                           int gameNumber) {
    GameReport rep;
    rep.nnueIsWhite = nnueIsWhite;

    chess::Board board;
    board.set960(false);

    ThreadInfo thread;

    const bool fromOpeningList = !openings.empty();

    if (fromOpeningList) {
        const Opening& op = openings[static_cast<size_t>(openingIndex) % openings.size()];
        board.setFen(op.fen);
        rep.startFen   = op.fen;
        rep.openingId  = op.id;
    } else {
        board.setFen(chess::constants::STARTPOS);
    }

    thread.accumulatorStack.resetAccumulators(board);

    clearSearchState();
    g_use_qs     = useQs;
    g_use_policy = opt.searchPolicy;
    g_silent     = true;

    // ---- Optional EPD best-move prefix ------------------------------------
    if (fromOpeningList && opt.epdPlayBm) {
        const Opening& op = openings[static_cast<size_t>(openingIndex) % openings.size()];

        if (!op.bm.empty()) {
            chess::Movelist legal;
            chess::movegen::legalmoves(legal, board);

            const chess::Move bm = chess::uci::uciToMove(board, op.bm);

            if (bm != chess::Move() && isLegalMove(legal, bm)) {
                rep.moves.push_back(chess::uci::moveToUci(bm));

                if (opt.verbose) {
                    std::cout << "  game " << gameNumber
                              << " ply " << std::right << std::setw(3) << int(rep.moves.size())
                              << "  " << std::left << std::setw(6) << "epd-bm"
                              << " (" << colorName(board.sideToMove()) << ") "
                              << chess::uci::moveToUci(bm)
                              << std::endl;
                }

                applyMove(board, thread, bm);
            }
        }
    }

    // ---- Random opening ---------------------------------------------------
    if (!fromOpeningList && opt.openPlies > 0) {
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

        if (nnueTurn && opt.nnueMode == NnueMode::Static) {
            const double t0 = nowMs();

            const StaticNnueResult sr = staticNnueBestMove(board, thread);

            const double dt = nowMs() - t0;

            rep.nnueMs    += dt;
            rep.nnueNodes += sr.nodes;
            // mover-relative -> white-relative (mover is the NNUE side here)
            rep.lastNnueScore = nnueIsWhite ? sr.bestMoveScore : -sr.bestMoveScore;

            move = sr.ok ? sr.move : legal[0];

            if (move == chess::Move() || !isLegalMove(legal, move))
                move = legal[0];

            if (opt.verbose) {
                std::ostringstream oss;
                oss << " static-eval";
                if (opt.nodes > 0) oss << " n" << opt.nodes;
                oss << " eval " << sr.bestEvalScore
                    << " move-score " << sr.bestMoveScore
                    << " nodes " << sr.nodes;
                moveInfo = oss.str();
            }
        } else if (nnueTurn) {
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
        rep.plies = int(rep.moves.size());   // includes any opening prefix

        if (opt.verbose) {
            std::cout << "  game " << gameNumber
                      << " ply " << std::right << std::setw(3) << rep.moves.size()
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

static MatchResult runMatch(const Options& opt,
                            const std::vector<Opening>& openings,
                            bool useQs) {
    MatchResult res;
    res.qs = useQs;

    std::cout << "=====================================================================\n";
    std::cout << " match: Kociolek NNUE  vs  policy net (no search)\n";
    std::cout << " nnue player       : "
              << (opt.nnueMode == NnueMode::Static
                      ? std::string("static NNUE output only, no search (1 eval per legal move)")
                      : ("search, depth " + std::to_string(opt.depth)))
              << "\n";
    if (opt.nnueMode == NnueMode::Search) {
        if (opt.nodes > 0)    std::cout << "                     nodes " << opt.nodes << " per move\n";
        if (opt.movetime > 0) std::cout << "                     movetime " << opt.movetime << "ms per move\n";
        std::cout << "                     search-policy " << (opt.searchPolicy ? "on" : "off") << "\n";
        std::cout << " quiescence search : " << (useQs ? "ON" : "OFF") << "\n";
    } else {
        std::cout << " quiescence search : " << (useQs ? "ON" : "OFF")
                  << " (unused: there is no search at all)\n";
    }
    std::cout << " policy player     : " << (opt.useSmallPolicy ? "small" : "big")
              << " net, " << (opt.policyTemp <= 0.0f
                                ? std::string("argmax")
                                : ("sample T=" + std::to_string(opt.policyTemp)))
              << "\n";
    std::cout << " openings          : ";
    if (openings.empty()) {
        std::cout << "random " << opt.openPlies << " plies from startpos\n";
    } else {
        std::cout << openings.size() << " positions from ";
        for (size_t i = 0; i < opt.openingsFiles.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << opt.openingsFiles[i];
        }
        std::cout << " (" << opt.openingsOrder << ")\n";
    }
    std::cout << " games             : " << opt.games << "\n";
    std::cout << "=====================================================================\n";
    std::cout.flush();

    // Two games per opening with colours swapped, so each opening is played
    // once as white and once as black by the NNUE player. The `openings` list
    // already holds exactly the sampled positions, in playing order.
    for (int g = 0; g < opt.games; ++g) {
        const bool nnueIsWhite = (g % 2 == 0);
        const size_t pair      = size_t(g / 2);
        const int  openingIdx  = openings.empty()
                                     ? 0
                                     : int(pair % openings.size());

        const double t0 = nowMs();
        GameReport rep = playGame(opt, nnueIsWhite, useQs, openingIdx, openings, g + 1);
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

        if (!rep.openingId.empty()) {
            std::cout << "  opening: " << rep.openingId << "\n";
        } else if (!rep.startFen.empty() && opt.verbose) {
            std::cout << "  opening fen: " << rep.startFen << "\n";
        }

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
            opt.nnueMode == NnueMode::Static
                ? std::string("Kociolek-NNUE static")
                : ("Kociolek-NNUE d" + std::to_string(opt.depth) +
                   (match.qs ? " qs=on" : " qs=off"));

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

        if (!g.openingId.empty()) {
            out << "[Opening \"" << g.openingId << "\"]\n";
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
  --nnue-mode M        search   = full alpha-beta search          (default)
                       static   = NNUE output only, no search: evaluate
                                  every legal move once and play the best
                                  one (on startpos this is ~20 nodes, i.e.
                                  exactly d1 without quiescence)
  --static             shorthand for --nnue-mode static
  --qs on|off|both     quiescence search switch (search mode only);
                       "both" plays the whole match twice       (default on)
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
  --openings <file>    opening book: FEN per line, EPD (uho.epd,
                       EPD with bm/id ops) or "FEN moves ..." lines.
                       May be repeated; files are concatenated.
                       Both colours of a pair get the same opening.
  --uho [file]         shorthand for --openings uho.epd
  --openings-order O   random = shuffle the book, no repeats until it
                       runs out (default) | seq = file order
  --epd-bm             play the EPD "bm" move first when it is a UCI move
  --open-plies N       random opening plies from startpos when no
                       --openings/--uho is given                   (default 6)
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
  ./arena --uho --games 100                  # UHO openings, both colours
  ./arena --static --uho                    # NNUE eval only vs policy net
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
        } else if (a == "--static") {
            opt.nnueMode = NnueMode::Static;
        } else if (a == "--nnue-mode") {
            const char* v = needValue(i, a); if (!v) return false;

            std::string m(v);
            std::transform(m.begin(), m.end(), m.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });

            if (m == "static" || m == "eval" || m == "d1" || m == "1") {
                opt.nnueMode = NnueMode::Static;
            } else if (m == "search" || m == "full") {
                opt.nnueMode = NnueMode::Search;
            } else {
                std::cerr << "error: --nnue-mode expects search|static" << std::endl;
                return false;
            }
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
            opt.openingsFiles.push_back(v);
        } else if (a == "--uho") {
            // Optional file name: --uho [file] / --uho=file
            std::string file = "uho.epd";
            bool looksLikePath = false;

            if (i + 1 < argc) {
                const std::string next = argv[i + 1];

                if (!next.empty() && next[0] != '-' &&
                    (next.size() > 4 && (next.substr(next.size() - 4) == ".epd" ||
                                         next.substr(next.size() - 4) == ".fen" ||
                                         next.substr(next.size() - 4) == ".txt"))) {
                    looksLikePath = true;
                }
            }

            if (looksLikePath) file = argv[++i];

            opt.openingsFiles.push_back(file);
        } else if (a.rfind("--uho=", 0) == 0) {
            opt.openingsFiles.push_back(a.substr(6));
        } else if (a == "--epd-bm") {
            opt.epdPlayBm = true;
        } else if (a == "--openings-order") {
            const char* v = needValue(i, a); if (!v) return false;

            std::string o(v);
            std::transform(o.begin(), o.end(), o.begin(),
                           [](unsigned char c) { return char(std::tolower(c)); });

            if (o != "random" && o != "seq" && o != "sequential") {
                std::cerr << "error: --openings-order expects random|seq" << std::endl;
                return false;
            }

            opt.openingsOrder = (o == "random") ? "random" : "seq";
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

    if (opt.nnueMode == NnueMode::Static) {
        if (opt.qs == QsMode::Both) {
            std::cout << "info string --nnue-mode static has no search, so --qs both "
                         "would play the same match twice; using qs on only\n";
            opt.qs = QsMode::On;
        }

        if (opt.nodes > 0)
            std::cout << "info string --nnue-mode static ignores --nodes "
                         "(exactly one NNUE evaluation per legal move)\n";
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

    // ---- Opening book -----------------------------------------------------
    std::vector<Opening> openings;

    for (std::string& path : opt.openingsFiles) {
        const std::string resolved = resolveBookPath(path);

        if (resolved.empty()) {
            std::cerr << "error: no opening book found for '" << path << "'\n"
                      << "       put uho.epd (e.g. UHO_4060_v1.epd from\n"
                      << "       github.com/official-stockfish/books) next to the binary,\n"
                      << "       run 'make uho' to download it, or pass --openings <file>\n";
            return 1;
        }

        if (resolved != path) {
            std::cout << "info string using opening book " << resolved << "\n";
            path = resolved;
        }
    }

    if (!opt.openingsFiles.empty()) {
        std::string err;
        size_t total = 0;

        const size_t pairsNeeded = size_t((opt.games + 1) / 2);

        if (!loadOpenings(opt.openingsFiles, pairsNeeded, opt.openingsOrder,
                          opt.seed, openings, total, err)) {
            std::cerr << "error: " << err << "\n";
            return 1;
        }

        if (openings.empty()) {
            std::cerr << "error: no openings found in the given file(s)\n";
            return 1;
        }

        std::cout << "info string book contains " << total << " openings, using "
                  << openings.size() << " (" << opt.openingsOrder << " order)\n";

        if (total < pairsNeeded) {
            std::cout << "info string note: " << pairsNeeded
                      << " opening pairs requested but the book only has " << total
                      << "; openings will repeat\n";
        }
    }

    std::vector<MatchResult> matches;

    if (opt.qs == QsMode::Both) {
        matches.push_back(runMatch(opt, openings, true));
        matches.push_back(runMatch(opt, openings, false));
    } else {
        matches.push_back(runMatch(opt, openings, opt.qs == QsMode::On));
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
