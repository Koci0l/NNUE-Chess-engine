# NNUE vs Policy arena

`arena` is a small standalone application that loads the NNUE network **and** the
policy network and lets the two play a match against each other. It has an
on/off switch for the **quiescence search**, can play from a real **opening
book** (`uho.epd`), and can run the NNUE side with **no search at all** (its raw
output only).

```
make arena
./arena --uho --games 20 --qs both
```

## Build

```bash
make arena            # builds ./arena (nets are loaded from disk at runtime)
make                  # builds the UCI engine as before
make uho              # downloads an opening book (UHO_4060_v1.epd)
make clean            # removes ./arena and the engine binary
```

The arena deliberately does **not** link `src/uci.cpp`, so it never pulls in the
embedded policy blobs (`src/policy_embed*.h`) — the network files are opened at
start-up instead. That keeps the build fast and lets you swap nets per run.

## The two players

| | NNUE player | Policy player |
|---|---|---|
| network | `quantised-v14.bin` (EVALFILE) | `Policy-big.bin` (POLICYFILE) |
| move choice | search, or NNUE output only (`--static`) | one forward pass, no search |
| policy use | move ordering / time management (`--search-policy`) | the move distribution itself |
| quiescence search | controlled by `--qs` (search mode only) | not applicable (no search) |

The NNUE player is the engine's own search (with `UsePolicy` on by default), so
the match answers "how much does the raw policy net lose against real search?".
The policy player takes `argmax(softmax(logits))` by default, or samples from
`policy^(1/T)` with `--policy-temp T` to make it less deterministic.

## The quiescence-search switch

Three ways to flip it:

```bash
./arena --qs on            # quiescence search enabled (default)
./arena --qs off           # leaves are scored with the static NNUE eval only
./arena --qs both          # plays the whole match twice and prints a comparison
```

In the engine (UCI):

```
option name UseQS type check default true
qson
qsoff
setoption name UseQS value false
```

`--qs both` prints a table like

```
================= quiescence search comparison =================
 qs           games   wins  losses  draws    score     elo
 on              20     16       3      1    82.5%    +266
 off             20     12       6      2    65.0%    +108

 score(qs=on) - score(qs=off) = +17.5%   (positive means quiescence search helps)
```

## NNUE output only (no search)

`--static` (or `--nnue-mode static`) turns the NNUE player into a pure evaluator:
for every legal move it plays the move, runs **one** NNUE forward pass, undoes it,
and finally plays the move with the best resulting score. No alpha-beta, no
pruning, no quiescence search, no policy ordering — this is "depth 1" without QS:

```
$ ./arena --static --verbose --open-plies 0
  game 1 ply   1  NNUE   (white) e2e4 static-eval eval 72 move-score 30 nodes 20
  game 1 ply   2  Policy (black) d7d5 (policy)
```

On the start position that is exactly **20 nodes** — one per legal move, as
expected. The node count follows the position (31 after 1.e4 d5). `--depth`,
`--nodes`, `--movetime`, `--search-policy` and `--qs` do not apply in this mode.

## Opening books (uho.epd)

Without a book the arena plays a short random opening from the start position
(`--open-plies`, default 6 plies, seeded by `--seed`). With a book:

```bash
make uho                            # downloads UHO_4060_v1.epd (~16 MB)
./arena --uho --games 100           # finds uho.epd / UHO*.epd automatically
./arena --openings uho.epd --games 100
./arena --uho UHO_Lichess_4852_v1.epd --games 1000
./arena --openings a.fen b.epd      # several books are concatenated
```

**Formats accepted** (auto-detected, one position per line):

| format | example |
|---|---|
| FEN | `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` |
| EPD / uho.epd | `rnbqkbnr/... w KQkq - bm e2e4; id "UHO_Lichess_4852_v1.epd:1";` |
| UCI start | `rnbqkbnr/... w KQkq - moves e2e4 e7e5` |

`hmvc` / `fmvn` EPD ops are honoured, `bm` can be played as the first move
(`--epd-bm`), and `id` is printed per game and written to the PGN as an `Opening`
tag. Blank lines, `#` comments and a UTF-8 BOM are ignored; unparsable lines are
skipped with a warning.

Notes:

* The official UHO books in `official-stockfish/books` are plain FEN files:
  246 759 positions in `UHO_4060_v1.epd`, 2 632 036 in `UHO_Lichess_4852_v1.epd`.
  Loading is streamed and only the positions the match needs are kept, so the
  175 MB Lichess book costs ~5 s and a couple of MB of RAM instead of ~400 MB.
* Every opening is played as a pair — once with the NNUE net as white, once as
  black — so colour bias cancels out.
* `--openings-order random` (default) is a uniform random sample, no repeats
  until the book is exhausted; `seq` takes positions in file order.
* Books are git-ignored (`*.epd`): use `make uho` or drop the file next to the
  binary.

## Options

```
Match
  --games N            games to play, colours alternate          (default 20)
  --nnue-mode M        search (default) | static (NNUE eval only, see above)
  --static             shorthand for --nnue-mode static
  --qs on|off|both     quiescence search switch (search mode only);
                       "both" plays the whole match twice       (default on)
  --depth N            NNUE search depth per move                 (default 8)
  --nodes N            NNUE node limit per move (0 = off)         (default 0)
  --movetime MS        NNUE time per move in ms, min 75 (0 = off)  (default 0)
  --search-policy on|off
                       NNUE player uses the policy net for move
                       ordering / time management                (default on)

Networks
  --nnue <file>        NNUE network file            (default quantised-v14.bin)
  --policy <file>      policy network file            (default Policy-big.bin)
  --policy-small       load the file as the small net (POLICY_SMALL_HL)
  --policy-temp T      policy player: 0 = argmax, >0 = sample     (default 0)

Game handling
  --openings <file>    opening book: FEN per line, EPD (uho.epd) or
                       "FEN moves ..." lines; repeatable
  --uho [file]         shorthand for --openings, auto-finds uho.epd /
                       UHO*.epd in the current directory
  --openings-order O   random (default) | seq
  --epd-bm             play the EPD "bm" move first when it is a UCI move
  --open-plies N       random opening plies when no book is given (default 6)
  --max-plies N        adjudicate a draw after N plies            (default 240)
  --adjudicate-cp N    >0: adjudicate on |NNUE eval| at max-plies (default 0)
  --hash MB            transposition table size                   (default 128)
  --seed N             random seed, controls openings             (default 20240215)

Output
  --pgn <file>         write the games as PGN (UCI long algebraic moves)
  --verbose            print every move
  --help               usage text
```

## Examples

```bash
# default match, 20 games, depth 8, quiescence on, random openings
./arena

# real opening book, NNUE eval only vs the policy net
make uho && ./arena --uho --static --games 100

# does quiescence search help? (same openings for both halves of the match)
./arena --uho --qs both --games 40 --depth 8

# fast smoke test with a node budget instead of a depth budget
./arena --games 4 --depth 64 --nodes 20000

# watch one game move by move, write the PGN
./arena --games 2 --depth 6 --verbose --pgn match.pgn

# weaker, more "human" policy opponent
./arena --games 20 --policy-temp 0.8
```

## Sample output

```
 match: Kociolek NNUE  vs  policy net (no search)
 nnue player       : search, depth 8
                     search-policy on
 quiescence search : ON
 policy player     : big net, argmax
 openings          : 2 positions from uho.epd (random)
 games             : 4
=====================================================================
game   1/4  NNUE=w Policy=b  1-0  (checkmate)  plies 61  0.4s  nnue_nodes 812233
...
 NNUE wins 16   policy wins 3   draws 1   (20 games)
 NNUE score 82.50%   Elo +266  [12, 520]
```

## Tips

* A raw policy net (no search) is much weaker than even a depth-1 search. For
  closer, more informative matches use `--depth 1`..`3`, a node budget, a small
  `--movetime`, `--static`, and/or give the policy some randomness with
  `--policy-temp`.
* `--qs both` is a comparison tool: with few games the difference is pure noise
  (look at the printed Elo interval). Use a few hundred games before drawing
  conclusions.
* The NNUE and policy nets must match the hidden sizes the binary was compiled
  for (`KOCIOLEK_POLICY_HL`, `POLICY_SMALL_HL`); a size mismatch is reported as
  a load error.
* `--verbose` shows every move, including the opening prefix.
