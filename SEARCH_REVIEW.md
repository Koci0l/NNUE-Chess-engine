# Search review — Kociolek 2.3

Everything below was reproduced on this branch. Two helper tools were added so the
claims can be re-checked:

```bash
# UCI/regression checks: legal bestmove, pv[0] == bestmove, "a search actually
# happened", deterministic bench.  Needs: pip install chess
python3 tools/uci_smoke_test.py ./Kociolek-2.2.exe

# Incremental NNUE accumulator vs full recompute (7200 random positions)
# (build line is in the file header)
./acc_selfcheck quantised-v14.bin
```

---

## 1. "Sometimes it plays a random move" — there was **no search at all**

`TimeManager::init()` for a `movetime` budget set

```cpp
soft_limit_ms = hard_limit_ms = movetime_ms - MOVE_OVERHEAD_MS;   // overhead = 50
```

For any `movetime <= 50` that is **negative**, so `should_stop()` was true on the
very first node. The root loop bailed out at the top of its first move

```cpp
if (tm.should_stop()) goto search_done;
```

and `search()` returned `best_move`, which at that point is still the
*initialisation* value: `rootPolicy.top_any` (the policy net's favourite move) or
just `moves[0]` — the first move in move-generation order. Nothing was searched,
no `info` line was printed, and the GUI got a move that the engine never looked at.

Reproduced on the old code (`go movetime 30`):

| position | played | searched |
|---|---|---|
| startpos | `g2g3` | nothing |
| Kiwipete | `e2a6` | nothing |
| `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w` | `b4f4` | nothing |

Same thing with a small `go nodes`, or with any clock so short that the hard limit
is already exceeded when `go` arrives.

**Fixed:**
* `movetime` budget is floored at `MIN_THINKING_TIME`, and `hard >= soft + MIN`.
* Depth 1 always runs to completion: `root_tm` is `nullptr` for depth 1 and the
  root only polls the clock for `depth > 1`. There is no longer any path that
  returns a move that was not searched.

## 2. "It searches one move but plays another" — aspiration fail-high wrote to the result

On a fail-high the code did

```cpp
} else if (score >= aspiration_beta && aspiration_beta < MATE_SCORE) {
    best_score = score;
    best_move  = depth_best_move;      // <-- committed mid-iteration
    delta *= 2;
    ...                                // re-search with a wider window
    if (tm.should_stop()) goto search_done;
```

`best_move`/`best_score` were overwritten with a **fail-high bound** from an
*unfinished* iteration. If the re-search with the widened window was then cut off
by the clock, `search()` returned that unverified move — while the last `info`
line still showed the previous move. That is exactly the reported symptom.

**Fixed:** `best_move`/`best_score` now only ever describe the last *completed*
iteration. Window management uses a separate `aspiration_center`, and move
ordering uses a separate `root_first_move`, so neither can leak into the result.

## 3. Castling was classified as a capture everywhere

In this chess library a castling move is encoded as `king_from -> rook_from`, so
`board.at(move.to())` is **our own rook**. Every `at(to) != NONE` "is this a
capture?" test therefore said *yes* to castling:

* `MovePicker::scoreCaptures()` put castling in the capture list (ordered right
  after real captures) and `scoreQuiets()` removed it from the quiet list — so no
  history score, no killer/counter-move, and it landed in a list that is sorted by
  "victim value * 10 - attacker value" (`900*10 - 20000` for king-takes-rook).
* `isQuietMove()` returned `false` → castling was routed into the **capture**
  history, the *noisy*-move SEE-pruning branch, and the "noisy" branch of the
  history/LMR bookkeeping.
* `extractCaptureInfo()` happily recorded a "KING captures ROOK on h1/a1" entry in
  the capture history.
* The root ordering treated it as a tactical move (`see_ge(...)` on a move whose
  target square holds our own rook).

**Fixed:** `isCastlingMove()` / `isTacticalMove()` in `types.h`, used consistently
in `types.h`, `movepick.cpp` and `search.cpp`. Castling is a quiet move.

## 4. Triple singular extension → tree explosion

```cpp
int ext = 1;
if (!is_pv_node && val + SE_DOUBLE_BIAS < singular_beta) ext += 1;
if (!is_pv_node && is_quiet_move && val + SE_TRIPLE_BIAS < singular_beta) ext += 1;
```

`new_depth = depth + in_check + ext - 1` could be **depth + 3** from a *half-depth
null-window verification search*. A single node grows the subtree by 3 plies and
the reported score/depth curve becomes non-monotonic. Stockfish caps this at 2.

**Fixed:** triple extension off (`SE_ALLOW_TRIPLE_EXTENSION = false`) and
`local_extension` is hard-capped at `MAX_EXTENSION_PER_NODE = 2`.

## 5. Razoring with a 700 cp margin at depth 3

```cpp
if (depth <= 3) { margin = depth == 3 ? 700 : ...; if (static_eval + margin <= alpha) ...}
```

"If my static eval is seven pawns below alpha, a capture-only search is good
enough" throws away exactly the quiet defensive resources that save lost
positions. That is a direct "the engine misses the only move" generator.

**Fixed:** razoring only at depth 1–2 (`RAZOR_MAX_DEPTH`).

## 6. Root LMR driven by the policy net (up to ±3 plies extra)

At the root, non-favourite quiet moves got

```cpp
float adj = -0.85f * rel;
adj = std::clamp(adj, -2.0f, 3.0f);      // then *= sharpness
reduction += int(std::lround(adj)) * LMR_SCALE;
```

on top of the LMR table entry. A move the policy net disliked could be searched
**5 plies shallower** than the move it liked; if the reduced null-window search
failed low it was never searched at full depth and could never be played. Two
consequences: (a) the root decision depends on the net, not on the search, (b) any
SPSA run over LMR parameters is partly tuning the net's taste.

**Fixed:** policy adjustment clamped to ±1 ply, total root reduction clamped to
`ROOT_LMR_MAX_REDUCTION = 2` plies.

## 7. `board.zobrist()` at every node

`getZobristHash()` called `board.zobrist()`, which **recomputes the key from
scratch** (loop over all pieces + magic lookups) — at every alpha-beta *and* every
quiescence node, plus once per PV move. The library already maintains the key
incrementally (`board.hash()`, asserted equal in the library's own `setFen`).

**Fixed:** `return board.hash();` — identical results (bench node count is
unchanged by this commit), ~5% higher bench NPS on this VM (503.9k → 527.4k
best-of-3), and relatively more in positions with many pieces.

## 8. Clock read at every node

`tm->should_stop()` (a `high_resolution_clock::now()`) was called for every single
node. Now polled every 1024 nodes; the root still polls between moves.

## 9. "IID" that is not IID

```cpp
// IID reduction when no TT move
if (!in_singular_search && !is_pv_node && depth >= 4 && tt_move == Move() && !in_check)
    depth--;
```

No IID search happens — nothing is searched to *produce* a TT move. The node is
simply searched one ply shallower, at every node in the tree that has no TT entry.
A large part of the tree is therefore searched at "reported depth − 1" (on top of
the usual reductions) and TT depths are shifted accordingly.

**Not changed by default** — it is now `FIRST_VISIT_DEPTH_REDUCTION` (currently
`false`). A/B this one, it is a real strength knob, not a bug, but it makes every
depth-dependent parameter (LMR, RFP, futility) mean something different than the
name suggests.

---

## What was *not* broken

* **The NNUE accumulator is fine.** `tools/acc_selfcheck` applies 7200 random
  moves (castling, en passant, promotions, king moves across the d/e file) with
  the exact incremental update path used by the search and compares against a full
  recompute: **0 mismatches**. The eval is not "jumping" because of stale
  accumulators.
* **The search is not grossly unsound.** A brute-force reference search (full
  width, no TT, no pruning, capture-only qsearch) gives **-245** for Kiwipete at
  depth 4 — exactly what the engine prints at depth 4.
* **Move generation / legality** is fine; the UCI smoke test checks the best move
  and the whole PV for legality in 12 positions with depth, node, movetime and
  clock limits.

## Measured effect of the fixes

| check | old | new |
|---|---|---|
| `go movetime 30` (old: unsearched move) | 12/12 positions failed | 0 failures |
| PV self-consistency, depth 10, 14 random positions (score, then score after playing its own move) | worst gap **319 cp** | worst gap **69 cp** |
| bench nodes | 5,175,523 | 5,275,826 |
| bench NPS (best of 3) | 503.9k | 527.4k |
| 400 self-play games, depth 7, new vs old | — | **52.5%** for the new build |

## Why SPSA went to hell

1. Every tuning game had a chance of playing a move that was **never searched**
   (§1) or was based on a **fail-high bound from an aborted iteration** (§2).
   That is pure noise, and it is correlated with time, which SPSA is very good at
   exploiting in the wrong direction.
2. Root LMR (§6) made the root move depend on the policy net, so LMR parameters
   were partly tuning the net's priors.
3. LMR interacts with the unsound razoring (§5) and triple extensions (§4):
   changing LMR changes which nodes reach the razor/extension paths, so LMR values
   "compensate" for those instead of doing their own job.
4. `FIRST_VISIT_DEPTH_REDUCTION` (§9) shifts the depth scale, so every
   depth-proportional margin (RFP, futility, ProbCut, LMR) is tuned against a
   skewed depth.

Concretely, for the next tuning run:
* run SPSA with `setoption name UsePolicy value false` (or with the policy path
  off) until the policy net itself has been validated in a match,
* tune with **fixed nodes** rather than the clock, and keep the hash history/TT
  reset per game (`ucinewgame`),
* tune one group at a time: (LMR base/divisor) → (RFP/futility/razor) →
  (SE/ProbCut), re-running `tools/uci_smoke_test.py` and a short match after each
  group,
* re-derive `SE_TRIPLE_BIAS`/`RAZOR_MARGIN_D3` — they no longer exist as tunables.

## Things I would look at next (not changed here)

* **LMR shape.** `+LMR_SCALE` for every non-PV node, `+LMR_SCALE` for
  `!improving`, `+1` for cut nodes on top of a table that already gives 2.7 plies
  at depth 10 → a late quiet move at depth 10 is searched at depth ~4. Consider
  writing the reduction in *plies* with a single `improving`/`cutNode` term and a
  cap of `depth / 2`.
* **Correction history** is updated from *bound* scores (`score - raw_static_eval`
  at beta cutoffs is a lower bound of the error), which biases it in one
  direction. Only update on nodes where the score is exact.
* `MovePicker::isValid()` does a linear scan over all legal moves for the TT move,
  both killers and the counter move at every node.
* `AccumulatorStack::push()` memcpy's 4 KB per node. A per-ply "needs refresh"
  scheme or lazy updates would be a few percent.
* ProbCut runs a *quiescence* probe before the reduced search; the qsearch
  fail-high is a weak gate and the second search is at `depth - 4`.
