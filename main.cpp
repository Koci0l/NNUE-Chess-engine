#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <cstring>
#include <cmath>
#include "chess.hpp"
#include "nnue.h"

constexpr int MATE_SCORE = 30000;
constexpr int DEFAULT_SEARCH_DEPTH = 5;
constexpr int CONTEMPT = 0;
constexpr int MAX_PLY = 100;

// ============= NODE COUNTER =============
struct SearchStats {
    uint64_t nodes;
    
    SearchStats() : nodes(0) {}
    
    void reset() {
        nodes = 0;
    }
};

// ============= BUTTERFLY HISTORY =============
struct ButterflyHistory {
    static constexpr int HISTORY_MAX = 16384;
    static constexpr int HISTORY_GRAVITY = HISTORY_MAX;
    
    int16_t table[64][64];  // [from][to]
    
    ButterflyHistory() {
        clear();
    }
    
    void clear() {
        std::memset(table, 0, sizeof(table));
    }
    
    int get(chess::Square from, chess::Square to) const {
        return table[from.index()][to.index()];
    }
    
    void update(chess::Square from, chess::Square to, int bonus) {
        int f = from.index();
        int t = to.index();
        
        int cur = table[f][t];
        
        int delta = bonus - (cur * std::abs(bonus)) / HISTORY_GRAVITY;
        int next = cur + delta;
        
        next = std::max(-HISTORY_MAX, std::min(HISTORY_MAX, next));
        table[f][t] = static_cast<int16_t>(next);
    }
    
    void age() {
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 64; ++j) {
                table[i][j] /= 2;
            }
        }
    }
    
    bool should_age() const {
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 64; ++j) {
                if (std::abs(table[i][j]) > (HISTORY_MAX * 7) / 8) {
                    return true;
                }
            }
        }
        return false;
    }
};

// ============= KILLER MOVES =============
struct KillerMoves {
    chess::Move killers[MAX_PLY][2];  // [ply][slot] - 2 killer moves per ply
    
    KillerMoves() {
        clear();
    }
    
    void clear() {
        for (int i = 0; i < MAX_PLY; ++i) {
            killers[i][0] = chess::Move();
            killers[i][1] = chess::Move();
        }
    }
    
    void store(int ply, chess::Move move) {
        // Don't store if already in slot 0
        if (move == killers[ply][0]) {
            return;
        }
        
        // Shift: slot 0 -> slot 1, new move -> slot 0
        killers[ply][1] = killers[ply][0];
        killers[ply][0] = move;
    }
    
    bool is_killer(int ply, chess::Move move) const {
        if (ply >= MAX_PLY) return false;
        return move == killers[ply][0] || move == killers[ply][1];
    }
    
    int get_killer_score(int ply, chess::Move move) const {
        if (ply >= MAX_PLY) return 0;
        
        if (move == killers[ply][0]) return 2;  // Primary killer
        if (move == killers[ply][1]) return 1;  // Secondary killer
        return 0;
    }
};

// Global tables
ButterflyHistory g_butterflyHistory;
KillerMoves g_killerMoves;

// ============= TRANSPOSITION TABLE GLOBALS =============
size_t TT_SIZE = 1 << 20;  // Default 1M entries (~24MB)
size_t TT_MASK = TT_SIZE - 1;

enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2 };

struct TTEntry {
    uint64_t key;
    int16_t depth;
    int16_t score;
    chess::Move best_move;
    TTFlag flag;
    
    TTEntry() : key(0), depth(-1), score(0), flag(TT_EXACT) {}
};

TTEntry* tt = nullptr;

void initTT(size_t mb) {
    // Convert MB to number of entries
    size_t bytes = mb * 1024 * 1024;
    size_t entries = bytes / sizeof(TTEntry);
    
    // Round down to power of 2
    size_t power = 1;
    while (power * 2 <= entries) {
        power *= 2;
    }
    
    TT_SIZE = power;
    TT_MASK = TT_SIZE - 1;
    
    if (tt != nullptr) {
        delete[] tt;
    }
    
    tt = new TTEntry[TT_SIZE];
    std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    
    std::cout << "info string Hash table initialized: " << mb << " MB (" << TT_SIZE << " entries)" << std::endl;
}

void clearTT() {
    if (tt != nullptr) {
        std::memset(tt, 0, TT_SIZE * sizeof(TTEntry));
    }
}

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

// ============= DRAW DETECTION =============
bool isDrawByRepetition(const chess::Board& board) {
    return board.isRepetition(1);
}

bool isDrawByFiftyMove(const chess::Board& board) {
    return board.halfMoveClock() >= 100;
}

int getDrawScore(int ply_from_root) {
    return CONTEMPT;
}

// ============= TIME MANAGEMENT =============
struct TimeManager {
    int time_left_ms;
    int increment_ms;
    int movetime_ms;
    
    int soft_limit_ms;
    int hard_limit_ms;
    
    std::chrono::high_resolution_clock::time_point start_time;
    
    chess::Move last_best_move;
    int stability_count;
    
    static constexpr int MOVE_OVERHEAD_MS = 50;
    static constexpr int MIN_THINKING_TIME = 10;
    
    TimeManager() 
        : time_left_ms(0), increment_ms(0), movetime_ms(0),
          soft_limit_ms(0), hard_limit_ms(0), stability_count(0) {}
    
    void init(int time_ms, int inc_ms, int movestogo, int fixed_movetime) {
        time_left_ms = time_ms;
        increment_ms = inc_ms;
        movetime_ms = fixed_movetime;
        stability_count = 0;
        
        start_time = std::chrono::high_resolution_clock::now();
        
        if (movetime_ms > 0) {
            soft_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;
            hard_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;
        } else if (time_left_ms > 0) {
            int moves_left = movestogo > 0 ? movestogo : 40;
            int base_time = time_left_ms / std::max(20, moves_left);
            int usable_increment = (increment_ms * 3) / 4;
            
            int optimal_time = base_time + usable_increment;
            
            soft_limit_ms = std::min(optimal_time * 3, time_left_ms / 3);
            hard_limit_ms = std::min(optimal_time * 5, time_left_ms - MOVE_OVERHEAD_MS);
            
            soft_limit_ms = std::max(MIN_THINKING_TIME, soft_limit_ms);
            hard_limit_ms = std::max(soft_limit_ms, hard_limit_ms);
        } else {
            soft_limit_ms = 999999;
            hard_limit_ms = 999999;
        }
    }
    
    int64_t elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    }
    
    bool should_stop() const {
        return elapsed_ms() >= hard_limit_ms;
    }
    
    bool should_continue_depth(int depth, double last_depth_ms) const {
        int64_t elapsed = elapsed_ms();
        
        if (elapsed >= soft_limit_ms) return false;
        
        double estimated_next = last_depth_ms * 4.0;
        
        if (elapsed + estimated_next > hard_limit_ms) return false;
        
        return true;
    }
    
    void update_stability(chess::Move best_move) {
        if (best_move == last_best_move) {
            stability_count++;
        } else {
            stability_count = 0;
        }
        last_best_move = best_move;
    }
};

// ============= ZOBRIST HASHING =============
uint64_t zobrist_piece[12][64];
uint64_t zobrist_side;
uint64_t zobrist_castling[16];
uint64_t zobrist_ep_file[8];

void initZobrist() {
    std::mt19937_64 rng(0x1337BEEF);
    
    for (int p = 0; p < 12; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
            zobrist_piece[p][sq] = rng();
        }
    }
    
    zobrist_side = rng();
    
    for (int i = 0; i < 16; ++i) {
        zobrist_castling[i] = rng();
    }
    
    for (int f = 0; f < 8; ++f) {
        zobrist_ep_file[f] = rng();
    }
}

uint64_t getZobristHash(const chess::Board& board) {
    uint64_t hash = 0;
    
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece p = board.at(chess::Square(sq));
        if (p != chess::Piece::NONE) {
            int piece_idx = static_cast<int>(p.type()) + (p.color() == chess::Color::WHITE ? 0 : 6);
            hash ^= zobrist_piece[piece_idx][sq];
        }
    }
    
    if (board.sideToMove() == chess::Color::BLACK) {
        hash ^= zobrist_side;
    }
    
    int castling_rights = 0;
    if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)) 
        castling_rights |= 1;
    if (board.castlingRights().has(chess::Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE)) 
        castling_rights |= 2;
    if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)) 
        castling_rights |= 4;
    if (board.castlingRights().has(chess::Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE)) 
        castling_rights |= 8;
    hash ^= zobrist_castling[castling_rights];
    
    if (board.enpassantSq() != chess::Square::underlying::NO_SQ) {
        int ep_file = static_cast<int>(board.enpassantSq().file());
        hash ^= zobrist_ep_file[ep_file];
    }
    
    return hash;
}

// ============= NNUE SCALING =============
inline int scaleNNUE(int raw_score) {
    return raw_score / 2;
}

// ============= TT FUNCTIONS =============
void storeTT(uint64_t key, int depth, int score, chess::Move best_move, TTFlag flag, int ply_from_root) {
    size_t index = key & TT_MASK;
    TTEntry& entry = tt[index];
    
    int stored_score = score;
    if (score >= MATE_SCORE - 100) {
        stored_score = score + ply_from_root;
    } else if (score <= -MATE_SCORE + 100) {
        stored_score = score - ply_from_root;
    }
    
    if (entry.key != key || entry.depth < depth) {
        entry.key = key;
        entry.depth = depth;
        entry.score = stored_score;
        entry.best_move = best_move;
        entry.flag = flag;
    }
}

bool probeTT(uint64_t key, int depth, int alpha, int beta, int& score, chess::Move& tt_move, int ply_from_root) {
    size_t index = key & TT_MASK;
    const TTEntry& entry = tt[index];
    
    if (entry.key != key) {
        tt_move = chess::Move();
        return false;
    }
    
    tt_move = entry.best_move;
    
    if (entry.depth >= depth) {
        int retrieved_score = entry.score;
        
        if (retrieved_score >= MATE_SCORE - 100) {
            retrieved_score -= ply_from_root;
        } else if (retrieved_score <= -MATE_SCORE + 100) {
            retrieved_score += ply_from_root;
        }
        
        if (entry.flag == TT_EXACT) {
            score = retrieved_score;
            return true;
        }
        if (entry.flag == TT_LOWER && retrieved_score >= beta) {
            score = beta;
            return true;
        }
        if (entry.flag == TT_UPPER && retrieved_score <= alpha) {
            score = alpha;
            return true;
        }
    }
    
    return false;
}

// ============= PV EXTRACTION =============
std::vector<chess::Move> extractPV(chess::Board board, int max_depth) {
    std::vector<chess::Move> pv;
    
    for (int i = 0; i < max_depth; ++i) {
        uint64_t hash = getZobristHash(board);
        size_t index = hash & TT_MASK;
        const TTEntry& entry = tt[index];
        
        if (entry.key != hash || entry.best_move == chess::Move()) {
            break;
        }
        
        chess::Move move = entry.best_move;
        
        chess::Movelist legal_moves;
        chess::movegen::legalmoves(legal_moves, board);
        
        bool is_legal = false;
        for (const auto& m : legal_moves) {
            if (m == move) {
                is_legal = true;
                break;
            }
        }
        
        if (!is_legal) break;
        
        pv.push_back(move);
        board.makeMove(move);
        
        chess::Movelist next_moves;
        chess::movegen::legalmoves(next_moves, board);
        if (next_moves.empty()) {
            break;
        }
    }
    
    return pv;
}

// ============= ACCUMULATOR UPDATE =============
void updateAccumulatorForMove(AccumulatorStack& accStack, chess::Board& board, const chess::Move& move) {
    auto moveType = move.typeOf();
    
    if (moveType == chess::Move::NORMAL) {
        chess::Piece piece = board.at(move.from());
        chess::Piece captured = board.at(move.to());
        
        if (captured != chess::Piece::NONE) {
            accStack.current().remove_piece(captured, move.to());
        }
        accStack.current().move_piece(piece, move.from(), move.to());
        
    } else if (moveType == chess::Move::PROMOTION) {
        chess::Piece pawn = board.at(move.from());
        chess::Piece captured = board.at(move.to());
        chess::Piece promotedPiece = chess::Piece(move.promotionType(), pawn.color());
        
        if (captured != chess::Piece::NONE) {
            accStack.current().remove_piece(captured, move.to());
        }
        accStack.current().remove_piece(pawn, move.from());
        accStack.current().add_piece(promotedPiece, move.to());
        
    } else if (moveType == chess::Move::ENPASSANT) {
        chess::Piece pawn = board.at(move.from());
        chess::Square capturedPawnSq = chess::Square(move.to().file(), move.from().rank());
        chess::Piece capturedPawn = board.at(capturedPawnSq);
        
        accStack.current().remove_piece(capturedPawn, capturedPawnSq);
        accStack.current().move_piece(pawn, move.from(), move.to());
        
    } else if (moveType == chess::Move::CASTLING) {
        chess::Piece pieces_before[64];
        for (int i = 0; i < 64; i++) {
            pieces_before[i] = board.at(chess::Square(i));
        }
        
        board.makeMove(move);
        
        for (int i = 0; i < 64; i++) {
            chess::Square sq(i);
            chess::Piece before = pieces_before[i];
            chess::Piece after = board.at(sq);
            
            if (before != after) {
                if (before != chess::Piece::NONE) {
                    accStack.current().remove_piece(before, sq);
                }
                if (after != chess::Piece::NONE) {
                    accStack.current().add_piece(after, sq);
                }
            }
        }
        
        board.unmakeMove(move);
    }
}

// ============= MOVE ORDERING WITH INCREMENTAL SELECTION =============
int pieceValue(chess::PieceType pt) {
    static const int values[] = {100, 320, 330, 500, 900, 20000};
    int idx = static_cast<int>(pt);
    if (idx >= 0 && idx < 6) return values[idx];
    return 0;
}

struct ScoredMove {
    chess::Move move;
    int score;
    
    ScoredMove() : score(0) {}
    ScoredMove(chess::Move m, int s) : move(m), score(s) {}
};

int scoreMoveForOrdering(const chess::Board& board, const chess::Move& move, 
                         const chess::Move& tt_move, int ply) {
    // TT move gets highest priority
    if (move == tt_move && tt_move != chess::Move()) {
        return 1000000;
    }
    
    // Promotions
    if (move.typeOf() == chess::Move::PROMOTION) {
        return 900000 + pieceValue(move.promotionType());
    }
    
    // Captures (MVV-LVA)
    chess::Piece captured = board.at(move.to());
    if (captured != chess::Piece::NONE) {
        int victimValue = pieceValue(captured.type());
        int attackerValue = pieceValue(board.at(move.from()).type());
        return 800000 + victimValue * 10 - attackerValue;
    }
    
    // En passant captures
    if (move.typeOf() == chess::Move::ENPASSANT) {
        return 800000 + 100;
    }
    
    // Killer moves (only for quiet moves)
    int killer_score = g_killerMoves.get_killer_score(ply, move);
    if (killer_score > 0) {
        return 700000 + killer_score * 1000;  // Primary killer > Secondary killer
    }
    
    // Quiet moves - use butterfly history
    int score = g_butterflyHistory.get(move.from(), move.to());
    
    // Small bonus for castling
    if (move.typeOf() == chess::Move::CASTLING) {
        score += 50;
    }
    
    return score;
}

// Score all moves once
std::vector<ScoredMove> scoreMoves(const chess::Movelist& moves, const chess::Board& board, 
                                   const chess::Move& tt_move = chess::Move(), int ply = 0) {
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());
    
    for (const auto& move : moves) {
        scored.emplace_back(move, scoreMoveForOrdering(board, move, tt_move, ply));
    }
    
    return scored;
}

// Pick the next best move (swap it to position 'current')
void pickNextMove(std::vector<ScoredMove>& moves, size_t current) {
    if (current >= moves.size()) return;
    
    size_t best_idx = current;
    int best_score = moves[current].score;
    
    for (size_t i = current + 1; i < moves.size(); ++i) {
        if (moves[i].score > best_score) {
            best_score = moves[i].score;
            best_idx = i;
        }
    }
    
    if (best_idx != current) {
        std::swap(moves[current], moves[best_idx]);
    }
}

// ============= QUIESCENCE SEARCH =============
int quiescence(chess::Board& board, int alpha, int beta, ThreadInfo& thread, int ply_from_root, SearchStats& stats) {
    stats.nodes++;
    
    int stand_pat = scaleNNUE(g_nnue.evaluate(board, thread));
    
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;
    
    chess::Movelist all_moves;
    chess::movegen::legalmoves(all_moves, board);
    
    chess::Movelist tactical_moves;
    for (const auto& move : all_moves) {
        if (board.at(move.to()) != chess::Piece::NONE || 
            move.typeOf() == chess::Move::PROMOTION ||
            move.typeOf() == chess::Move::ENPASSANT) {
            tactical_moves.add(move);
        }
    }
    
    auto scored_moves = scoreMoves(tactical_moves, board, chess::Move(), ply_from_root);
    
    for (size_t i = 0; i < scored_moves.size(); ++i) {
        pickNextMove(scored_moves, i);
        const auto& move = scored_moves[i].move;
        
        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);
        
        int eval = -quiescence(board, -beta, -alpha, thread, ply_from_root + 1, stats);
        
        board.unmakeMove(move);
        thread.accumulatorStack.pop();
        
        if (eval >= beta) return beta;
        if (eval > alpha) alpha = eval;
    }
    
    return alpha;
}

// ============= HELPER FUNCTIONS =============
bool isQuietMove(const chess::Board& board, const chess::Move& move) {
    return board.at(move.to()) == chess::Piece::NONE && 
           move.typeOf() != chess::Move::PROMOTION &&
           move.typeOf() != chess::Move::ENPASSANT;
}

bool hasNonPawnMaterial(const chess::Board& board) {
    chess::Color side = board.sideToMove();
    
    if (board.pieces(chess::PieceType::KNIGHT, side).count() > 0) return true;
    if (board.pieces(chess::PieceType::BISHOP, side).count() > 0) return true;
    if (board.pieces(chess::PieceType::ROOK, side).count() > 0) return true;
    if (board.pieces(chess::PieceType::QUEEN, side).count() > 0) return true;
    
    return false;
}

// ============= LMR REDUCTION TABLE =============
int lmr_reductions[64][64];

void initLMR() {
    for (int depth = 1; depth < 64; ++depth) {
        for (int move_num = 1; move_num < 64; ++move_num) {
            double reduction = 0.75 + std::log(depth) * std::log(move_num) / 2.25;
            lmr_reductions[depth][move_num] = static_cast<int>(reduction);
        }
    }
}

// ============= ALPHA-BETA SEARCH =============
int alphaBeta(chess::Board& board, int depth, int alpha, int beta, int ply_from_root, 
              ThreadInfo& thread, const TimeManager* tm, SearchStats& stats, bool allow_null) {
    stats.nodes++;
    
    if (tm && tm->should_stop()) return alpha;
    
    if (ply_from_root > 0) {
        if (isDrawByRepetition(board)) return getDrawScore(ply_from_root);
        if (isDrawByFiftyMove(board)) return getDrawScore(ply_from_root);
    }
    
    uint64_t hash = getZobristHash(board);
    int tt_score = 0;
    chess::Move tt_move;
    
    if (probeTT(hash, depth, alpha, beta, tt_score, tt_move, ply_from_root)) {
        return tt_score;
    }
    
    bool in_check = board.inCheck();
    bool is_pv_node = (beta - alpha) > 1;
    
    int static_eval = 0;
    if (!in_check) {
        static_eval = scaleNNUE(g_nnue.evaluate(board, thread));
    }
    
    // Check extension
    int extension = 0;
    if (in_check) {
        extension = 1;
    }
    
    // Reverse futility pruning
    if (!is_pv_node &&
        !in_check &&
        depth <= 7 &&
        depth >= 1 &&
        std::abs(beta) < MATE_SCORE - 100) {
        
        int rfp_margin = 85 * depth;
        
        if (static_eval - rfp_margin >= beta) {
            return static_eval - rfp_margin;
        }
    }
    
    // Null move pruning
    if (allow_null && 
        !in_check && 
        !is_pv_node &&
        depth >= 3 &&
        hasNonPawnMaterial(board)) {
        
        constexpr int R = 2;
        
        AccumulatorPair saved_acc = thread.accumulatorStack.current();
        
        board.makeNullMove();
        
        thread.accumulatorStack.push();
        thread.accumulatorStack.current() = saved_acc;
        
        int null_score = -alphaBeta(board, depth - R - 1, -beta, -beta + 1, 
                                    ply_from_root + 1, thread, tm, stats, false);
        
        thread.accumulatorStack.pop();
        board.unmakeNullMove();
        
        if (null_score >= beta) {
            if (null_score >= MATE_SCORE - 100) {
                return beta;
            }
            return null_score;
        }
    }
    
    if (depth + extension <= 0) {
        return quiescence(board, alpha, beta, thread, ply_from_root, stats);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        if (in_check) return -MATE_SCORE + ply_from_root;
        return getDrawScore(ply_from_root);
    }
    
    auto scored_moves = scoreMoves(moves, board, tt_move, ply_from_root);
    
    chess::Move best_move;
    int best_score = -MATE_SCORE;
    int original_alpha = alpha;
    
    std::vector<chess::Move> quiets_searched;
    
    int move_count = 0;

    for (size_t i = 0; i < scored_moves.size(); ++i) {
        pickNextMove(scored_moves, i);
        const auto& move = scored_moves[i].move;
        
        move_count++;
        
        bool is_quiet = isQuietMove(board, move);
        
        // Late move pruning
        if (!is_pv_node &&
            !in_check &&
            depth <= 8 &&
            is_quiet &&
            move != tt_move &&
            move_count >= 5 + 2 * depth * depth) {
            continue;
        }
        
        thread.accumulatorStack.push();
        updateAccumulatorForMove(thread.accumulatorStack, board, move);
        board.makeMove(move);
        
        bool gives_check = board.inCheck();
        
        // Futility pruning
        if (!is_pv_node &&
            !in_check &&
            !gives_check &&
            depth <= 7 &&
            is_quiet &&
            move != tt_move &&
            std::abs(alpha) < MATE_SCORE - 100) {
            
            int futility_margin = 400 + 120 * depth;
            
            if (static_eval + futility_margin <= alpha) {
                board.unmakeMove(move);
                thread.accumulatorStack.pop();
                continue;
            }
        }
        
        int eval;
        int new_depth = depth + extension - 1;
        
        // LMR
        bool can_reduce = !in_check &&
                         is_quiet &&
                         move_count > 1 &&
                         depth >= 3 &&
                         !gives_check;
        
        if (can_reduce) {
            int reduction = lmr_reductions[std::min(depth, 63)][std::min(move_count, 63)];
            
            if (move == tt_move) {
                reduction = 0;
            } else if (move_count <= 3) {
                reduction = std::max(0, reduction - 1);
            }
            
            reduction = std::max(1, std::min(reduction, new_depth - 1));
            
            eval = -alphaBeta(board, new_depth - reduction, -alpha - 1, -alpha, 
                            ply_from_root + 1, thread, tm, stats, true);
            
            if (eval > alpha) {
                eval = -alphaBeta(board, new_depth, -beta, -alpha, 
                                ply_from_root + 1, thread, tm, stats, true);
            }
        } else {
            if (move_count == 1) {
                eval = -alphaBeta(board, new_depth, -beta, -alpha, 
                                ply_from_root + 1, thread, tm, stats, true);
            } else {
                eval = -alphaBeta(board, new_depth, -alpha - 1, -alpha, 
                                ply_from_root + 1, thread, tm, stats, true);
                
                if (eval > alpha && eval < beta) {
                    eval = -alphaBeta(board, new_depth, -beta, -alpha, 
                                    ply_from_root + 1, thread, tm, stats, true);
                }
            }
        }
        
        board.unmakeMove(move);
        thread.accumulatorStack.pop();
        
        if (eval > best_score) {
            best_score = eval;
            best_move = move;
        }

        if (eval >= beta) {
            // *** KILLER MOVE UPDATE - Store beta cutoff quiet moves ***
            if (is_quiet) {
                g_killerMoves.store(ply_from_root, move);
                
                int bonus = 32 * depth * depth;
                
                g_butterflyHistory.update(move.from(), move.to(), bonus);
                
                for (const auto& quiet : quiets_searched) {
                    g_butterflyHistory.update(quiet.from(), quiet.to(), -bonus / 2);
                }
            }
            
            storeTT(hash, depth, beta, best_move, TT_LOWER, ply_from_root);
            return beta;
        }
        
        if (eval > alpha) {
            alpha = eval;
        }
        
        if (is_quiet) {
            quiets_searched.push_back(move);
        }
    }
    
    if (best_score > original_alpha && isQuietMove(board, best_move)) {
        int bonus = 8 * depth * depth;
        
        g_butterflyHistory.update(best_move.from(), best_move.to(), bonus);
        
        for (const auto& quiet : quiets_searched) {
            if (quiet != best_move) {
                g_butterflyHistory.update(quiet.from(), quiet.to(), -bonus / 4);
            }
        }
    }
    
    TTFlag flag = (best_score > original_alpha) ? TT_EXACT : TT_UPPER;
    storeTT(hash, depth, best_score, best_move, flag, ply_from_root);
    
    return best_score;
}

// ============= ROOT SEARCH WITH ASPIRATION WINDOWS =============
chess::Move search(chess::Board& board, int max_depth, ThreadInfo& thread, TimeManager& tm) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) return chess::Move();
    if (moves.size() == 1) {
        std::cout << "info string only move" << std::endl;
        return moves[0];
    }

    if (g_butterflyHistory.should_age()) {
        g_butterflyHistory.age();
        std::cout << "info string butterfly history aged" << std::endl;
    }

    chess::Move best_move = moves[0];
    int best_score = -MATE_SCORE;
    double last_depth_ms = 100.0;
    
    auto scored_moves = scoreMoves(moves, board);

    for (int depth = 1; depth <= max_depth; ++depth) {
        auto depth_start = std::chrono::high_resolution_clock::now();
        
        if (depth > 1 && !tm.should_continue_depth(depth, last_depth_ms)) {
            std::cout << "info string stopping at depth " << (depth - 1) 
                     << " (time: " << tm.elapsed_ms() << "ms)" << std::endl;
            break;
        }

        SearchStats stats;
        stats.reset();
        
        int alpha, beta;
        int delta = 25;
        chess::Move depth_best_move = scored_moves[0].move;
        int score;
        
        bool search_again = true;
        while (search_again) {
            search_again = false;
            
            if (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100) {
                alpha = std::max(-MATE_SCORE, best_score - delta);
                beta = std::min(MATE_SCORE, best_score + delta);
            } else {
                alpha = -MATE_SCORE - 1;
                beta = MATE_SCORE + 1;
            }
            
            stats.reset();
            
            // Re-score moves with current TT move
            scored_moves = scoreMoves(moves, board);
            
            score = -MATE_SCORE;
            
            for (size_t i = 0; i < scored_moves.size(); ++i) {
                pickNextMove(scored_moves, i);
                const auto& move = scored_moves[i].move;
                
                if (tm.should_stop()) {
                    std::cout << "info string time limit reached at depth " << depth << std::endl;
                    goto search_done;
                }
                
                thread.accumulatorStack.push();
                updateAccumulatorForMove(thread.accumulatorStack, board, move);
                board.makeMove(move);
                
                bool is_draw_move = isDrawByRepetition(board) || isDrawByFiftyMove(board);
                int eval;
                
                if (is_draw_move) {
                    eval = -getDrawScore(1);
                } else {
                    eval = -alphaBeta(board, depth - 1, -beta, -alpha, 1, thread, &tm, stats, true);
                }
                
                board.unmakeMove(move);
                thread.accumulatorStack.pop();

                if (eval > score) {
                    score = eval;
                    depth_best_move = move;
                }
                
                if (eval > alpha) {
                    alpha = eval;
                }
                
                if (eval >= beta) {
                    break;
                }
            }
            
            if (score <= (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100 ? 
                         std::max(-MATE_SCORE, best_score - delta) : -MATE_SCORE - 1)) {
                std::cout << "info string depth " << depth << " failed low (" 
                         << score << "), widening window" << std::endl;
                delta *= 2;
                search_again = true;
                
            } else if (score >= (depth >= 5 && std::abs(best_score) < MATE_SCORE - 100 ? 
                                std::min(MATE_SCORE, best_score + delta) : MATE_SCORE + 1)) {
                std::cout << "info string depth " << depth << " failed high (" 
                         << score << "), widening window" << std::endl;
                delta *= 2;
                search_again = true;
            }
            
            if (delta > 1000) {
                std::cout << "info string depth " << depth << " window too wide, using full window" << std::endl;
                delta = MATE_SCORE;
            }
            
            if (search_again && tm.should_stop()) {
                std::cout << "info string time limit reached during aspiration re-search at depth " << depth << std::endl;
                goto search_done;
            }
        }

        best_score = score;
        best_move = depth_best_move;
        
        auto depth_end = std::chrono::high_resolution_clock::now();
        last_depth_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            depth_end - depth_start).count();
        
        int64_t elapsed = tm.elapsed_ms();
        uint64_t nps = (elapsed > 0) ? (stats.nodes * 1000) / elapsed : 0;

        auto pv_line = extractPV(board, depth);
        
        std::string pv_str;
        for (const auto& m : pv_line) {
            pv_str += chess::uci::moveToUci(m) + " ";
        }
        if (pv_str.empty()) {
            pv_str = chess::uci::moveToUci(best_move);
        }
        
        std::string score_str;
        if (best_score >= MATE_SCORE - 100) {
            int mate_in = (MATE_SCORE - best_score + 1) / 2;
            score_str = "mate " + std::to_string(mate_in);
        } else if (best_score <= -MATE_SCORE + 100) {
            int mate_in = -(MATE_SCORE + best_score) / 2;
            score_str = "mate " + std::to_string(mate_in);
        } else {
            score_str = "cp " + std::to_string(best_score);
        }

        std::cout << "info score " << score_str
                  << " depth " << depth 
                  << " nodes " << stats.nodes
                  << " nps " << nps
                  << " time " << elapsed 
                  << " pv " << pv_str << std::endl;

        tm.update_stability(best_move);
    }

search_done:
    return best_move;
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

void uci_loop() {
    chess::Board board;
    ThreadInfo thread;

    initZobrist();
    initLMR();
    initTT(64);
    
    std::cout << "info string Loading NNUE..." << std::endl;
    g_nnue.loadNetwork("quantised-v2.bin");
    std::cout << "info string NNUE loaded" << std::endl;
    std::cout << "info string Features: Incremental NNUE + QS + Pick-Best Move Ordering + TT + Butterfly History + Killer Moves + LMR + NMP + PV + Check Extensions + RFP + LMP + Futility + Aspiration Windows" << std::endl;
    
    board.setFen(chess::constants::STARTPOS);
    thread.accumulatorStack.resetAccumulators(board);

    std::string line;
    while (std::getline(std::cin, line)) {
        auto tokens = split(line, ' ');
        if (tokens.empty()) continue;

        if (tokens[0] == "uci") {
            std::cout << "id name MyNNUEEngine v11.0-Killers" << std::endl;
            std::cout << "id author Kociolek" << std::endl;
            std::cout << "option name Hash type spin default 64 min 1 max 1024" << std::endl;
            std::cout << "uciok" << std::endl;
            
        } else if (tokens[0] == "setoption") {
            if (tokens.size() >= 5 && tokens[1] == "name" && tokens[2] == "Hash" && tokens[3] == "value") {
                int hash_mb = std::stoi(tokens[4]);
                hash_mb = std::max(1, std::min(1024, hash_mb));
                initTT(hash_mb);
            }
            
        } else if (tokens[0] == "isready") {
            std::cout << "readyok" << std::endl;
            
        } else if (tokens[0] == "eval") {
            g_nnue.showBuckets(&board, thread.accumulatorStack.current());
            
        } else if (tokens[0] == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
            thread.accumulatorStack.resetAccumulators(board);
            clearTT();
            g_butterflyHistory.clear();
            g_killerMoves.clear();  // *** Clear killer moves for new game ***
            
        } else if (tokens[0] == "position") {
            int move_start_index = -1;
            std::string fen_str;
            
            if (tokens.size() >= 2 && tokens[1] == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                if (tokens.size() > 2 && tokens[2] == "moves") {
                    move_start_index = 3;
                }
            } else if (tokens.size() >= 2 && tokens[1] == "fen") {
                int i = 2;
                for (; i < (int)tokens.size(); ++i) {
                    if (tokens[i] == "moves") {
                        move_start_index = i + 1;
                        break;
                    }
                    fen_str += tokens[i] + " ";
                }
                board.setFen(fen_str);
            }

            thread.accumulatorStack.resetAccumulators(board);

            if (move_start_index != -1) {
                for (int i = move_start_index; i < (int)tokens.size(); ++i) {
                    chess::Move move = chess::uci::uciToMove(board, tokens[i]);
                    board.makeMove(move);
                    thread.accumulatorStack.resetAccumulators(board);
                }
            }
            
        } else if (tokens[0] == "go") {
            int depth = 100;
            int movetime = 0;
            int wtime = 0, btime = 0, winc = 0, binc = 0;
            int movestogo = 0;

            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "depth" && i + 1 < tokens.size() && is_integer(tokens[i + 1])) {
                    depth = std::max(1, std::stoi(tokens[i + 1]));
                    i++;
                } else if (tokens[i] == "movetime" && i + 1 < tokens.size()) {
                    movetime = std::stoi(tokens[i + 1]);
                    i++;
                } else if (tokens[i] == "wtime" && i + 1 < tokens.size()) {
                    wtime = std::stoi(tokens[i + 1]);
                    i++;
                } else if (tokens[i] == "btime" && i + 1 < tokens.size()) {
                    btime = std::stoi(tokens[i + 1]);
                    i++;
                } else if (tokens[i] == "winc" && i + 1 < tokens.size()) {
                    winc = std::stoi(tokens[i + 1]);
                    i++;
                } else if (tokens[i] == "binc" && i + 1 < tokens.size()) {
                    binc = std::stoi(tokens[i + 1]);
                    i++;
                } else if (tokens[i] == "movestogo" && i + 1 < tokens.size()) {
                    movestogo = std::stoi(tokens[i + 1]);
                    i++;
                } else if (tokens[i] == "infinite") {
                    depth = 100;
                }
            }
            
            if (tokens.size() == 2 && is_integer(tokens[1])) {
                depth = std::max(1, std::stoi(tokens[1]));
            }
            
            TimeManager tm;
            int mytime = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
            int myinc = (board.sideToMove() == chess::Color::WHITE) ? winc : binc;
            tm.init(mytime, myinc, movestogo, movetime);

            chess::Move best_move = search(board, depth, thread, tm);
            std::cout << "bestmove " << chess::uci::moveToUci(best_move) << std::endl;

        } else if (tokens[0] == "quit") {
            break;
        }
    }
    
    if (tt != nullptr) {
        delete[] tt;
    }   
}

int main(int argc, char* argv[]) {
    uci_loop();
    return 0;
}