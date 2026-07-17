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
PolicyRankCache g_policy_rank_cache;

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

static inline float crelu01(float x) {
    if (x < 0.f) return 0.f;
    if (x > 1.f) return 1.f;
    return x;
}

#ifndef POLICY_CASTLE_SEE_FORCE
#define POLICY_CASTLE_SEE_FORCE -1
#endif

// ============================================================================
// PolicyRankCache
// ============================================================================

void PolicyRankCache::clear() {
    for (int i = 0; i < POLICY_RANK_CACHE_SIZE; ++i) {
        table[i].key = ~0ULL;
        table[i].n = 0;
        table[i].nq = 0;
    }
    hits = 0;
    misses = 0;
}

bool PolicyRankCache::probe(uint64_t hash, int nmoves, int* out_ranks, int* out_nq) {
    if (nmoves <= 0 || nmoves > POLICY_RANK_CACHE_MAX_MOVES) return false;
    Entry& e = table[hash & (POLICY_RANK_CACHE_SIZE - 1)];
    if (e.key != hash || e.n != nmoves) {
        ++misses;
        return false;
    }
    for (int i = 0; i < nmoves; ++i) out_ranks[i] = static_cast<int>(e.ranks[i]);
    if (out_nq) *out_nq = e.nq;
    ++hits;
    return true;
}

void PolicyRankCache::store(uint64_t hash, int nmoves, const int* ranks, int nq) {
    if (nmoves <= 0 || nmoves > POLICY_RANK_CACHE_MAX_MOVES) return;
    Entry& e = table[hash & (POLICY_RANK_CACHE_SIZE - 1)];
    e.key = hash;
    e.n = static_cast<int16_t>(nmoves);
    e.nq = static_cast<int16_t>(nq);
    for (int i = 0; i < nmoves; ++i) {
        int r = ranks[i];
        if (r < -1) r = -1;
        if (r > 127) r = 127;
        e.ranks[i] = static_cast<int8_t>(r);
    }
}

bool policyRanksForNode(const chess::Board& board, uint64_t hash,
                        const chess::Movelist& moves,
                        int* out_rank, int* out_nq) {
    const int n = static_cast<int>(moves.size());
    if (out_nq) *out_nq = 0;
    if (!g_policy.loaded || n <= 0) return false;

    if (g_policy_rank_cache.probe(hash, n, out_rank, out_nq)) {
        return true;
    }

    if (!g_policy.rankLegalQuiets(board, moves, out_rank, out_nq)) {
        return false;
    }

    g_policy_rank_cache.store(hash, n, out_rank, out_nq ? *out_nq : 0);
    return true;
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
              << " mode=root_lmr+tm_v1+prune"
              << " lmr_top=" << POLICY_ROOT_LMR_TOP
              << " tm_min_depth=" << POLICY_TM_MIN_DEPTH
              << " prune_max_depth=" << POLICY_PRUNE_MAX_DEPTH
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

    bool good_see = false;

#if POLICY_CASTLE_SEE_FORCE >= 0
    if (is_castle) {
        good_see = (POLICY_CASTLE_SEE_FORCE != 0);
    } else
#endif
    if (is_castle) {
#if POLICY_CASTLE_SEE_FORCE < 0
        const chess::Square from_sq = m.from();
        const chess::Square to_sq(static_cast<chess::Square::underlying>(to_b));
        const chess::Move king_walk = chess::Move::make(from_sq, to_sq);
        good_see = chess::see::see_ge(board, king_walk, POLICY_SEE_TH);
#endif
    } else {
        good_see = chess::see::see_ge(board, m, POLICY_SEE_TH);
    }

    const int index = from_to * static_cast<int>(good_see) + idx;
    if (index < 0 || index >= num_moves) {
        return -1;
    }
    return index;
}

// ============================================================================
// Hidden
// ============================================================================

static void computeHidden(const PolicyNet& net,
                          const int* feats, int nfeats,
                          float* h1) {
    float h0[POLICY_HL];
    std::memcpy(h0, net.l0b.data(), sizeof(float) * POLICY_HL);

    for (int i = 0; i < nfeats; ++i) {
        const int f = feats[i];
        if (f < 0 || f >= POLICY_INPUT_SIZE) continue;
        const float* row = net.l0w.data() + size_t(f) * size_t(POLICY_HL);
        for (int j = 0; j < POLICY_HL; ++j) {
            h0[j] += row[j];
        }
    }

    for (int i = 0; i < POLICY_HL_PAIR; ++i) {
        h1[i] = crelu01(h0[i]) * crelu01(h0[i + POLICY_HL_PAIR]);
    }
}

static float logitForMoveIndex(const PolicyNet& net, const float* h1, int mi) {
    float logit = net.l1b[static_cast<size_t>(mi)];
    if (net.l1_out_major) {
        const float* row = net.l1w.data() + size_t(mi) * size_t(POLICY_HL_PAIR);
        for (int k = 0; k < POLICY_HL_PAIR; ++k) {
            logit += h1[k] * row[k];
        }
    } else {
        for (int k = 0; k < POLICY_HL_PAIR; ++k) {
            logit += h1[k] * net.l1w[size_t(k) * size_t(net.num_moves) + size_t(mi)];
        }
    }
    return logit;
}

static bool isQuietMoveLocal(const chess::Board& board, const chess::Move& m) {
    if (m.typeOf() == chess::Move::PROMOTION) return false;
    if (m.typeOf() == chess::Move::ENPASSANT) return false;
    if (board.at(m.to()) != chess::Piece::NONE) return false;
    return true;
}

// ============================================================================
// Forward
// ============================================================================

bool PolicyNet::logitsLegalMoves(const chess::Board& board,
                                 const chess::Movelist& moves,
                                 float* out_logits) const {
    if (!loaded || moves.empty()) {
        return false;
    }

    int feats[POLICY_MAX_ACTIVE];
    int nfeats = 0;
    collectFeatures(board, feats, nfeats);

    float h1[POLICY_HL_PAIR];
    computeHidden(*this, feats, nfeats, h1);

    for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
        const int mi = mapMoveToIndex(board, moves[i]);
        if (mi < 0) {
            out_logits[i] = -1e9f;
            continue;
        }
        out_logits[i] = logitForMoveIndex(*this, h1, mi);
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

bool PolicyNet::rankLegalQuiets(const chess::Board& board,
                                const chess::Movelist& moves,
                                int* out_rank,
                                int* out_nq) const {
    const int n = static_cast<int>(moves.size());
    for (int i = 0; i < n; ++i) {
        out_rank[i] = -1;
    }
    if (out_nq) *out_nq = 0;

    if (!loaded || n <= 0) {
        return false;
    }

    int quiet_i[256];
    float logits[256];
    int nq = 0;

    for (int i = 0; i < n; ++i) {
        if (!isQuietMoveLocal(board, moves[i])) continue;
        if (nq >= 256) break;
        quiet_i[nq++] = i;
    }

    if (out_nq) *out_nq = nq;
    if (nq <= 0) {
        return true;
    }

    int feats[POLICY_MAX_ACTIVE];
    int nfeats = 0;
    collectFeatures(board, feats, nfeats);

    float h1[POLICY_HL_PAIR];
    computeHidden(*this, feats, nfeats, h1);

    for (int q = 0; q < nq; ++q) {
        const int i = quiet_i[q];
        const int mi = mapMoveToIndex(board, moves[i]);
        logits[q] = (mi >= 0) ? logitForMoveIndex(*this, h1, mi) : -1e9f;
    }

    int order[256];
    for (int q = 0; q < nq; ++q) order[q] = q;
    std::sort(order, order + nq, [&](int a, int b) {
        return logits[a] > logits[b];
    });

    for (int rank = 0; rank < nq; ++rank) {
        const int q = order[rank];
        out_rank[quiet_i[q]] = rank;
    }

    return true;
}

bool PolicyNet::rootAdvice(const chess::Board& board,
                           chess::Move& out_top,
                           float& out_top1_prob,
                           float* entropy_out) const {
    out_top = chess::Move();
    out_top1_prob = 0.f;
    if (entropy_out) *entropy_out = 0.f;

    if (!loaded) return false;

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) return false;

    float probs_stack[256];
    std::vector<float> probs_heap;
    float* probs = probs_stack;
    if (moves.size() > 256) {
        probs_heap.resize(moves.size());
        probs = probs_heap.data();
    }

    if (!scoreLegalMoves(board, moves, probs)) return false;

    int best_i = 0;
    float best_p = probs[0];
    double ent = 0.0;
    for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
        if (probs[i] > best_p) {
            best_p = probs[i];
            best_i = i;
        }
        if (probs[i] > 1e-12f) {
            ent -= double(probs[i]) * std::log(double(probs[i]));
        }
    }

    out_top = moves[best_i];
    out_top1_prob = best_p;
    if (entropy_out) *entropy_out = static_cast<float>(ent);
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
              << " mode=root_lmr+tm_v1+prune"
              << std::endl;

    std::cout << "info string features (" << nfeats << "):";
    for (int i = 0; i < nfeats; ++i) std::cout << " " << feats[i];
    std::cout << std::endl;

    {
        float h1[POLICY_HL_PAIR];
        computeHidden(*this, feats, nfeats, h1);
        std::cout << "info string h1[0..7]:";
        for (int i = 0; i < 8; ++i) std::cout << " " << h1[i];
        std::cout << std::endl;
    }

    std::vector<float> logits(moves.size(), 0.f);
    std::vector<float> probs(moves.size(), 0.f);
    std::vector<int> ranks(moves.size(), -1);
    int nq = 0;

    if (!logitsLegalMoves(board, moves, logits.data())) {
        std::cout << "info string logits failed" << std::endl;
        return;
    }
    scoreLegalMoves(board, moves, probs.data());
    rankLegalQuiets(board, moves, ranks.data(), &nq);

    std::vector<int> order(moves.size());
    for (size_t i = 0; i < moves.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] > probs[b];
    });

    const int nshow = std::min(topN, static_cast<int>(moves.size()));
    std::cout << "info string rank  move     idx   see  logit      prob   qrank" << std::endl;

    for (int r = 0; r < nshow; ++r) {
        const int i = order[r];
        const chess::Move& m = moves[i];
        const int mi = mapMoveToIndex(board, m);
        bool good_see = false;
        if (m.typeOf() == chess::Move::CASTLING) {
            const bool ks = m.to() > m.from();
            const int kto = chess::Square::castling_king_square(ks, board.at(m.from()).color()).index();
            const chess::Move kw = chess::Move::make(m.from(), chess::Square(static_cast<chess::Square::underlying>(kto)));
            good_see = chess::see::see_ge(board, kw, POLICY_SEE_TH);
        } else {
            good_see = chess::see::see_ge(board, m, POLICY_SEE_TH);
        }

        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "info string #%02d  %-6s  %5d  %s  %+9.4f  %6.2f%%  %5d",
                      r + 1,
                      chess::uci::moveToUci(m).c_str(),
                      mi,
                      good_see ? "Y" : "N",
                      logits[i],
                      probs[i] * 100.f,
                      ranks[i]);
        std::cout << buf << std::endl;
    }

    double ent = 0.0;
    for (float p : probs) {
        if (p > 1e-12f) ent -= double(p) * std::log(double(p));
    }

    std::cout << "info string entropy=" << ent
              << " top1=" << (probs[order[0]] * 100.f) << "%"
              << " legal=" << moves.size()
              << " nquiets=" << nq
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

    bool good_see = false;
    if (m.typeOf() == chess::Move::CASTLING) {
        const bool ks = m.to() > m.from();
        const int kto = chess::Square::castling_king_square(ks, board.at(m.from()).color()).index();
        const chess::Move kw = chess::Move::make(
            m.from(), chess::Square(static_cast<chess::Square::underlying>(kto)));
        good_see = chess::see::see_ge(board, kw, POLICY_SEE_TH);
    } else {
        good_see = chess::see::see_ge(board, m, POLICY_SEE_TH);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    float logit = -1e9f;
    float prob  = 0.f;
    int   rank  = -1;
    int   qrank = -1;
    int   nq    = 0;

    if (!moves.empty()) {
        std::vector<float> logits(moves.size(), 0.f);
        std::vector<float> probs(moves.size(), 0.f);
        std::vector<int> ranks(moves.size(), -1);
        logitsLegalMoves(board, moves, logits.data());
        scoreLegalMoves(board, moves, probs.data());
        rankLegalQuiets(board, moves, ranks.data(), &nq);

        for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
            if (moves[i] == m) {
                logit = logits[i];
                prob  = probs[i];
                qrank = ranks[i];
                break;
            }
        }

        int better = 0;
        for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
            if (probs[i] > prob) ++better;
        }
        rank = better + 1;
    }

    if (m.typeOf() == chess::Move::CASTLING) {
        const int ksq = stmKingIndex(board);
        const int hm  = ((ksq % 8) > 3) ? 7 : 0;
        const bool ks = m.to() > m.from();
        const int is_ks = ks ? 1 : 0;
        const int is_hm = (hm == 0) ? 1 : 0;
        const int cidx = offsets[5][64] + POLICY_PROMOS + (is_ks ^ is_hm);
        std::cout << "info string castle_channels idx0=" << cidx
                  << " idx1=" << (cidx + from_to)
                  << " chosen=" << mi
                  << " see=" << (good_see ? 1 : 0)
                  << std::endl;
    }

    std::cout << "info string move " << chess::uci::moveToUci(m)
              << " idx=" << mi
              << " see=" << (good_see ? 1 : 0)
              << " logit=" << logit
              << " prob=" << (prob * 100.f) << "%"
              << " qrank=" << qrank
              << "/" << nq
              << " rank=" << rank
              << "/" << moves.size()
              << std::endl;
    std::cout.flush();
}