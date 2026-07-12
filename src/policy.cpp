#include "policy.h"
#include "policy_embed.h"
#include "see.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

PolicyNet g_policy;

// ============================================================================
// Bit helpers
// ============================================================================

static int popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0;
    while (x) {
        x &= x - 1;
        ++c;
    }
    return c;
#endif
}

static int ctz64_local(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx);
#else
    int n = 0;
    while ((x & 1ULL) == 0ULL) {
        x >>= 1;
        ++n;
    }
    return n;
#endif
}

static uint64_t bswap64_local(uint64_t x) {
    x = ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
    x = ((x & 0x00FF00FF00FF00FFULL) << 8)  | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
    return x;
}

// ============================================================================
// Destination tables (must match inputs.rs)
// ============================================================================

static constexpr uint64_t FILE_A = 0x0101010101010101ULL;
static constexpr uint64_t FILE_H = FILE_A << 7;

static constexpr uint64_t DIAGS[15] = {
    0x0100000000000000ULL, 0x0201000000000000ULL, 0x0402010000000000ULL,
    0x0804020100000000ULL, 0x1008040201000000ULL, 0x2010080402010000ULL,
    0x4020100804020100ULL, 0x8040201008040201ULL, 0x0080402010080402ULL,
    0x0000804020100804ULL, 0x0000008040201008ULL, 0x0000000080402010ULL,
    0x0000000000804020ULL, 0x0000000000008040ULL, 0x0000000000000080ULL,
};

static uint64_t destPawn(int sq) {
    const uint64_t bit = 1ULL << sq;
    return ((bit & ~FILE_A) << 7) | (bit << 8) | ((bit & ~FILE_H) << 9);
}

static uint64_t destKnight(int sq) {
    const uint64_t n = 1ULL << sq;
    const uint64_t h1 = ((n >> 1) & 0x7f7f7f7f7f7f7f7fULL) | ((n << 1) & 0xfefefefefefefefeULL);
    const uint64_t h2 = ((n >> 2) & 0x3f3f3f3f3f3f3f3fULL) | ((n << 2) & 0xfcfcfcfcfcfcfcfcULL);
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

static uint64_t destBishop(int sq) {
    const int rank = sq / 8;
    const int file = sq % 8;
    return bswap64_local(DIAGS[file + rank]) ^ DIAGS[7 + file - rank];
}

static uint64_t destRook(int sq) {
    const int rank = sq / 8;
    const int file = sq % 8;
    return (0xFFULL << (rank * 8)) ^ (FILE_A << file);
}

static uint64_t destQueen(int sq) {
    return destBishop(sq) | destRook(sq);
}

static uint64_t destKing(int sq) {
    uint64_t k = 1ULL << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & ~FILE_A) >> 1) | ((k & ~FILE_H) << 1);
    return k ^ (1ULL << sq);
}

// ============================================================================
// PolicyNet
// ============================================================================

PolicyNet::PolicyNet() {
    for (int sq = 0; sq < 64; ++sq) {
        destinations[sq][0] = destPawn(sq);
        destinations[sq][1] = destKnight(sq);
        destinations[sq][2] = destBishop(sq);
        destinations[sq][3] = destRook(sq);
        destinations[sq][4] = destQueen(sq);
        destinations[sq][5] = destKing(sq);
    }

    int curr = 0;
    for (int pc = 0; pc < 6; ++pc) {
        for (int sq = 0; sq < 64; ++sq) {
            offsets[pc][sq] = curr;
            curr += popcount64(destinations[sq][pc]);
        }
        offsets[pc][64] = curr;
    }

    from_to   = offsets[5][64] + POLICY_PROMOS + 2 + 8;
    num_moves = 2 * from_to;
    l1_out_major = true;
}

void PolicyNet::clear() {
    loaded = false;
    l0w.clear();
    l0b.clear();
    l1w.clear();
    l1b.clear();
}

bool PolicyNet::loadFromMemory(const std::uint8_t* data, std::size_t size, const char* label) {
    clear();

    if (!data || size == 0) {
        std::cerr << "info string Policy: empty memory blob ("
                  << (label ? label : "?") << ")" << std::endl;
        return false;
    }

    const size_t expected =
        size_t(POLICY_INPUT_SIZE) * size_t(POLICY_HL) +
        size_t(POLICY_HL) +
        size_t(POLICY_HL_PAIR) * size_t(num_moves) +
        size_t(num_moves);

    if (size != expected) {
        std::cerr << "info string Policy: size mismatch got " << size
                  << " expected " << expected
                  << " (from_to=" << from_to
                  << " num_moves=" << num_moves
                  << " label=" << (label ? label : "?") << ")"
                  << std::endl;
        return false;
    }

    auto read_q = [&](std::vector<float>& dst, size_t n, size_t& off) -> bool {
        if (off + n > size) return false;
        dst.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const int8_t q = static_cast<int8_t>(data[off++]);
            dst[i] = float(q) / float(POLICY_QA);
        }
        return true;
    };

    size_t off = 0;
    if (!read_q(l0w, size_t(POLICY_INPUT_SIZE) * size_t(POLICY_HL), off) ||
        !read_q(l0b, size_t(POLICY_HL), off) ||
        !read_q(l1w, size_t(POLICY_HL_PAIR) * size_t(num_moves), off) ||
        !read_q(l1b, size_t(num_moves), off) ||
        off != size) {
        std::cerr << "info string Policy: truncated memory read" << std::endl;
        clear();
        return false;
    }

    l1_out_major = true;
    loaded = true;

    std::cout << "info string Policy loaded " << (label ? label : "memory")
              << " input=" << POLICY_INPUT_SIZE
              << " hl=" << POLICY_HL
              << " from_to=" << from_to
              << " moves=" << num_moves
              << " l1_out_major=1"
              << " bytes=" << size
              << std::endl;
    return true;
}

bool PolicyNet::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "info string Policy: failed to open " << path << std::endl;
        return false;
    }

    in.seekg(0, std::ios::end);
    const auto sz = static_cast<size_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> buf(sz);
    if (sz > 0) {
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
        if (!in) {
            std::cerr << "info string Policy: failed reading " << path << std::endl;
            return false;
        }
    }

    return loadFromMemory(buf.data(), buf.size(), path.c_str());
}

// ============================================================================
// Board helpers
// ============================================================================

int PolicyNet::stmKingIndex(const chess::Board& board) {
    return board.kingSq(board.sideToMove()).index();
}

int PolicyNet::flipMask(const chess::Board& board) {
    const int ksq  = stmKingIndex(board);
    const int vert = (board.sideToMove() == chess::Color::BLACK) ? 56 : 0;
    const int hori = ((ksq % 8) > 3) ? 7 : 0;
    return vert ^ hori;
}

uint64_t PolicyNet::attacksBySide(const chess::Board& board, chess::Color side) {
    using namespace chess;

    uint64_t threats = 0;
    const Bitboard occ = board.occ();

    {
        uint64_t bb = board.pieces(PieceType::PAWN, side).getBits();
        while (bb) {
            const int sq = ctz64_local(bb);
            bb &= bb - 1;
            threats |= attacks::pawn(side, Square(sq)).getBits();
        }
    }
    {
        uint64_t bb = board.pieces(PieceType::KNIGHT, side).getBits();
        while (bb) {
            const int sq = ctz64_local(bb);
            bb &= bb - 1;
            threats |= attacks::knight(Square(sq)).getBits();
        }
    }
    {
        uint64_t bb = board.pieces(PieceType::BISHOP, side).getBits();
        while (bb) {
            const int sq = ctz64_local(bb);
            bb &= bb - 1;
            threats |= attacks::bishop(Square(sq), occ).getBits();
        }
    }
    {
        uint64_t bb = board.pieces(PieceType::ROOK, side).getBits();
        while (bb) {
            const int sq = ctz64_local(bb);
            bb &= bb - 1;
            threats |= attacks::rook(Square(sq), occ).getBits();
        }
    }
    {
        uint64_t bb = board.pieces(PieceType::QUEEN, side).getBits();
        while (bb) {
            const int sq = ctz64_local(bb);
            bb &= bb - 1;
            threats |= attacks::queen(Square(sq), occ).getBits();
        }
    }

    threats |= attacks::king(board.kingSq(side)).getBits();
    return threats;
}

// ============================================================================
// Features
// ============================================================================

void PolicyNet::collectFeatures(const chess::Board& board, int* feats, int& nfeats) const {
    nfeats = 0;

    const int flip = flipMask(board);
    const chess::Color stm  = board.sideToMove();
    const chess::Color nstm = ~stm;

    const uint64_t threats  = attacksBySide(board, nstm);
    const uint64_t defences = attacksBySide(board, stm);

    static const chess::PieceType kPts[6] = {
        chess::PieceType::PAWN,   chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK,   chess::PieceType::QUEEN,  chess::PieceType::KING
    };

    for (int p = 0; p < 6; ++p) {
        const int pc = 64 * p;

        uint64_t ours = board.pieces(kPts[p], stm).getBits();
        while (ours) {
            const int sq = ctz64_local(ours);
            ours &= ours - 1;

            int feat = pc + (sq ^ flip);
            const uint64_t bit = 1ULL << sq;
            if (threats & bit)  feat += POLICY_PLANE;
            if (defences & bit) feat += POLICY_PLANE * 2;

            if (nfeats < POLICY_MAX_ACTIVE) {
                feats[nfeats++] = feat;
            }
        }

        uint64_t opps = board.pieces(kPts[p], nstm).getBits();
        while (opps) {
            const int sq = ctz64_local(opps);
            opps &= opps - 1;

            int feat = 384 + pc + (sq ^ flip);
            const uint64_t bit = 1ULL << sq;
            if (threats & bit)  feat += POLICY_PLANE;
            if (defences & bit) feat += POLICY_PLANE * 2;

            if (nfeats < POLICY_MAX_ACTIVE) {
                feats[nfeats++] = feat;
            }
        }
    }
}

// ============================================================================
// Move index
// ============================================================================

int PolicyNet::mapMoveToIndex(const chess::Board& board, const chess::Move& m) const {
    const int ksq = stmKingIndex(board);
    const int hm  = ((ksq % 8) > 3) ? 7 : 0;
    const int flip = hm ^ ((board.sideToMove() == chess::Color::BLACK) ? 56 : 0);

    const chess::Piece moved = board.at(m.from());
    if (moved == chess::Piece::NONE) {
        return -1;
    }

    const bool is_castle = (m.typeOf() == chess::Move::CASTLING);
    const bool is_promo  = (m.typeOf() == chess::Move::PROMOTION);
    const bool is_ep     = (m.typeOf() == chess::Move::ENPASSANT);

    const bool is_dbl =
        !is_castle && !is_promo && !is_ep &&
        moved.type() == chess::PieceType::PAWN &&
        m.from().file() == m.to().file() &&
        std::abs(static_cast<int>(m.from().rank()) - static_cast<int>(m.to().rank())) == 2;

    int from_b = m.from().index();
    int to_b   = m.to().index();

    bool king_side = false;
    if (is_castle) {
        king_side = (m.to() > m.from());
        const chess::Color c = moved.color();
        to_b = chess::Square::castling_king_square(king_side, c).index();
    }

    const int src = from_b ^ flip;
    const int dst = to_b ^ flip;

    int idx = 0;

    if (is_promo) {
        const int ffile = src % 8;
        const int tfile = dst % 8;
        const int promo_id = 2 * ffile + tfile;

        int promo_pc = 3;
        const chess::PieceType pt = m.promotionType();
        if (pt == chess::PieceType::KNIGHT)      promo_pc = 0;
        else if (pt == chess::PieceType::BISHOP) promo_pc = 1;
        else if (pt == chess::PieceType::ROOK)   promo_pc = 2;
        else if (pt == chess::PieceType::QUEEN)  promo_pc = 3;

        idx = offsets[5][64] + (POLICY_PROMOS / 4) * promo_pc + promo_id;
    } else if (is_castle) {
        const int is_ks = king_side ? 1 : 0;
        const int is_hm = (hm == 0) ? 1 : 0;
        idx = offsets[5][64] + POLICY_PROMOS + (is_ks ^ is_hm);
    } else if (is_dbl) {
        idx = offsets[5][64] + POLICY_PROMOS + 2 + (src % 8);
    } else {
        const int pc = static_cast<int>(moved.type());
        if (pc < 0 || pc > 5) {
            return -1;
        }
        const uint64_t dest_bb = destinations[src][pc];
        const uint64_t below = dest_bb & ((1ULL << dst) - 1ULL);
        idx = offsets[pc][src] + popcount64(below);
    }

    const bool good_see = chess::see::see_ge(board, m, POLICY_SEE_TH);
    const int index = from_to * static_cast<int>(good_see) + idx;

    if (index < 0 || index >= num_moves) {
        return -1;
    }
    return index;
}

// ============================================================================
// Forward
// ============================================================================

static inline float crelu01(float x) {
    if (x < 0.f) return 0.f;
    if (x > 1.f) return 1.f;
    return x;
}

bool PolicyNet::logitsLegalMoves(const chess::Board& board,
                                 const chess::Movelist& moves,
                                 float* out_logits) const {
    if (!loaded || moves.empty()) {
        return false;
    }

    int feats[POLICY_MAX_ACTIVE];
    int nfeats = 0;
    collectFeatures(board, feats, nfeats);

    float h0[POLICY_HL];
    std::memcpy(h0, l0b.data(), sizeof(float) * POLICY_HL);

    for (int i = 0; i < nfeats; ++i) {
        const int f = feats[i];
        if (f < 0 || f >= POLICY_INPUT_SIZE) continue;
        const float* row = l0w.data() + size_t(f) * size_t(POLICY_HL);
        for (int j = 0; j < POLICY_HL; ++j) {
            h0[j] += row[j];
        }
    }

    float h1[POLICY_HL_PAIR];
    for (int i = 0; i < POLICY_HL_PAIR; ++i) {
        h1[i] = crelu01(h0[i]) * crelu01(h0[i + POLICY_HL_PAIR]);
    }

    for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
        const int mi = mapMoveToIndex(board, moves[i]);
        if (mi < 0) {
            out_logits[i] = -1e9f;
            continue;
        }

        float logit = l1b[static_cast<size_t>(mi)];

        if (l1_out_major) {
            const float* row = l1w.data() + size_t(mi) * size_t(POLICY_HL_PAIR);
            for (int k = 0; k < POLICY_HL_PAIR; ++k) {
                logit += h1[k] * row[k];
            }
        } else {
            for (int k = 0; k < POLICY_HL_PAIR; ++k) {
                logit += h1[k] * l1w[size_t(k) * size_t(num_moves) + size_t(mi)];
            }
        }

        out_logits[i] = logit;
    }

    return true;
}

bool PolicyNet::scoreLegalMoves(const chess::Board& board,
                                const chess::Movelist& moves,
                                float* out_probs) const {
    if (!logitsLegalMoves(board, moves, out_probs)) {
        return false;
    }

    const int n = static_cast<int>(moves.size());
    float mx = out_probs[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, out_probs[i]);

    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        out_probs[i] = std::exp(out_probs[i] - mx);
        sum += out_probs[i];
    }

    const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (int i = 0; i < n; ++i) out_probs[i] *= inv;
    return true;
}

// ============================================================================
// Debug
// ============================================================================

void PolicyNet::debugPosition(const chess::Board& board, int topN) const {
    if (!loaded) {
        std::cout << "info string Policy not loaded" << std::endl;
        return;
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) {
        std::cout << "info string no legal moves" << std::endl;
        return;
    }

    int feats[POLICY_MAX_ACTIVE];
    int nfeats = 0;
    collectFeatures(board, feats, nfeats);

    const int flip = flipMask(board);
    const int ksq  = stmKingIndex(board);

    std::cout << "info string === POLICY DEBUG ===" << std::endl;
    std::cout << "info string fen: " << board.getFen() << std::endl;
    std::cout << "info string stm: "
              << (board.sideToMove() == chess::Color::WHITE ? "w" : "b")
              << " king=" << ksq
              << " flip=" << flip
              << " from_to=" << from_to
              << " num_moves=" << num_moves
              << " l1_out_major=" << (l1_out_major ? 1 : 0)
              << std::endl;

    std::cout << "info string features (" << nfeats << "):";
    for (int i = 0; i < nfeats; ++i) std::cout << " " << feats[i];
    std::cout << std::endl;

    {
        float h0[POLICY_HL];
        std::memcpy(h0, l0b.data(), sizeof(float) * POLICY_HL);
        for (int i = 0; i < nfeats; ++i) {
            const int f = feats[i];
            if (f < 0 || f >= POLICY_INPUT_SIZE) continue;
            const float* row = l0w.data() + size_t(f) * size_t(POLICY_HL);
            for (int j = 0; j < POLICY_HL; ++j) h0[j] += row[j];
        }
        float h1[POLICY_HL_PAIR];
        for (int i = 0; i < POLICY_HL_PAIR; ++i) {
            h1[i] = crelu01(h0[i]) * crelu01(h0[i + POLICY_HL_PAIR]);
        }

        std::cout << "info string h0[0..7]:";
        for (int i = 0; i < 8; ++i) std::cout << " " << h0[i];
        std::cout << std::endl;

        std::cout << "info string h1[0..7]:";
        for (int i = 0; i < 8; ++i) std::cout << " " << h1[i];
        std::cout << std::endl;
    }

    std::vector<float> logits(moves.size(), 0.f);
    std::vector<float> probs(moves.size(), 0.f);

    if (!logitsLegalMoves(board, moves, logits.data())) {
        std::cout << "info string logits failed" << std::endl;
        return;
    }

    probs = logits;
    float mx = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) mx = std::max(mx, probs[i]);
    float sum = 0.f;
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] = std::exp(probs[i] - mx);
        sum += probs[i];
    }
    const float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (size_t i = 0; i < probs.size(); ++i) probs[i] *= inv;

    std::vector<int> order(moves.size());
    for (size_t i = 0; i < moves.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] > probs[b];
    });

    const int nshow = std::min(topN, static_cast<int>(moves.size()));
    std::cout << "info string rank  move     idx   see  logit      prob" << std::endl;

    for (int r = 0; r < nshow; ++r) {
        const int i = order[r];
        const chess::Move& m = moves[i];
        const int mi = mapMoveToIndex(board, m);
        const bool good_see = chess::see::see_ge(board, m, POLICY_SEE_TH);

        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "info string #%02d  %-6s  %5d  %s  %+9.4f  %6.2f%%",
                      r + 1,
                      chess::uci::moveToUci(m).c_str(),
                      mi,
                      good_see ? "Y" : "N",
                      logits[i],
                      probs[i] * 100.f);
        std::cout << buf << std::endl;
    }

    if (topN >= static_cast<int>(moves.size())) {
        std::cout << "info string --- all moves (uci idx see logit prob) ---" << std::endl;
        for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
            const int mi = mapMoveToIndex(board, moves[i]);
            const bool good_see = chess::see::see_ge(board, moves[i], POLICY_SEE_TH);
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "info string   %-6s  %5d  %s  %+9.4f  %6.2f%%",
                          chess::uci::moveToUci(moves[i]).c_str(),
                          mi,
                          good_see ? "Y" : "N",
                          logits[i],
                          probs[i] * 100.f);
            std::cout << buf << std::endl;
        }
    }

    double ent = 0.0;
    for (float p : probs) {
        if (p > 1e-12f) ent -= double(p) * std::log(double(p));
    }

    float logit_min = logits[0], logit_max = logits[0], logit_sum = 0.f;
    for (float l : logits) {
        logit_min = std::min(logit_min, l);
        logit_max = std::max(logit_max, l);
        logit_sum += l;
    }

    std::cout << "info string entropy=" << ent
              << " top1=" << (probs[order[0]] * 100.f) << "%"
              << " legal=" << moves.size()
              << std::endl;
    std::cout << "info string logits min=" << logit_min
              << " max=" << logit_max
              << " mean=" << (logit_sum / float(logits.size()))
              << std::endl;
    std::cout << "info string === END POLICY DEBUG ===" << std::endl;
    std::cout.flush();
}

void PolicyNet::debugMove(const chess::Board& board, const chess::Move& m) const {
    if (!loaded) {
        std::cout << "info string Policy not loaded" << std::endl;
        return;
    }

    const int mi = mapMoveToIndex(board, m);
    const bool good_see = chess::see::see_ge(board, m, POLICY_SEE_TH);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    float logit = -1e9f;
    float prob  = 0.f;
    int   rank  = -1;

    const chess::Piece moved = board.at(m.from());
    const bool is_castle = (m.typeOf() == chess::Move::CASTLING);
    const bool is_promo  = (m.typeOf() == chess::Move::PROMOTION);
    const bool is_ep     = (m.typeOf() == chess::Move::ENPASSANT);
    const bool is_dbl =
        !is_castle && !is_promo && !is_ep &&
        moved != chess::Piece::NONE &&
        moved.type() == chess::PieceType::PAWN &&
        m.from().file() == m.to().file() &&
        std::abs(static_cast<int>(m.from().rank()) - static_cast<int>(m.to().rank())) == 2;

    if (!moves.empty()) {
        std::vector<float> logits(moves.size(), 0.f);
        std::vector<float> probs(moves.size(), 0.f);
        logitsLegalMoves(board, moves, logits.data());
        scoreLegalMoves(board, moves, probs.data());

        for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
            if (moves[i] == m) {
                logit = logits[i];
                prob  = probs[i];
                break;
            }
        }

        int better = 0;
        for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
            if (probs[i] > prob) ++better;
        }
        rank = better + 1;
    }

    std::cout << "info string move " << chess::uci::moveToUci(m)
              << " from=" << m.from().index()
              << " to=" << m.to().index()
              << " type="
              << (is_castle ? "castle" :
                  is_promo  ? "promo"  :
                  is_ep     ? "ep"     :
                  is_dbl    ? "dbl"    : "normal")
              << " piece=" << (moved == chess::Piece::NONE ? -1 : static_cast<int>(moved.type()))
              << std::endl;

    std::cout << "info string   idx=" << mi
              << " see=" << (good_see ? 1 : 0)
              << " logit=" << logit
              << " prob=" << (prob * 100.f) << "%"
              << " rank=" << rank
              << "/" << moves.size()
              << std::endl;
    std::cout.flush();
}