# NNUE vs Policy arena

`arena` is a small standalone application that loads the NNUE network **and** the
policy network and lets the two play a match against each other. It also has an
on/off switch for the **quiescence search**.

```
make arena
./arena --games 20 --qs both
```

## Build

```bash
make arena            # builds ./arena (nets are loaded from disk at runtime)
make                  # builds the UCI engine as before
make clean            # removes ./arena and the engine binary
```

The arena deliberately does **not** link `src/uci.cpp`, so it never pulls in the
embedded policy blobs (`src/policy_embed*.h`) — the network files are opened at
start-up instead. That keeps the build fast and lets you swap nets per run.

## The two players

| | NNUE player | Policy player |
|---|---|---|
| network | `quantised-v14.bin` (EVALFILE) | `Policy-big.bin` (POLICYFILE) |
| move choice | full alpha-beta search (`search()`) | one forward pass, no search |
| policy use | move ordering / time management (`--search-policy`) | the move distribution itself |
| quiescence search | controlled by `--qs` | not applicable (no search) |

The NNUE player is exactly the engine's search with `UsePolicy` on, so the match
answers "how much does the raw policy net lose against real search?".
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

## Options

```
Match
  --games N            games to play, colours alternate          (default 20)
  --qs on|off|both     quiescence search switch                   (default on)
  --depth N            NNUE search depth per move                 (default 8)
  --nodes N            NNUE node limit per move (0 = off)         (default 0)
  --movetime MS        NNUE time per move in ms (0 = off)         (default 0)
  --search-policy on|off
                       NNUE player uses the policy net for move
                       ordering / time management                (default on)

Networks
  --nnue <file>        NNUE network file            (default quantised-v14.bin)
  --policy <file>      policy network file            (default Policy-big.bin)
  --policy-small       load the file as the small net (POLICY_SMALL_HL)
  --policy-temp T      policy player: 0 = argmax, >0 = sample     (default 0)

Game handling
  --open-plies N       random opening plies from startpos         (default 6)
  --openings <file>    one FEN per line instead of random openings
  --max-plies N        adjudicate a draw after N plies            (default 240)
  --adjudicate-cp N    >0: adjudicate on |NNUE eval| at max-plies (default 0)
  --hash MB            transposition table size                   (default 128)
  --seed N             random seed, controls openings             (default 20240215)

Output
  --pgn <file>         write the games as PGN (UCI long algebraic moves)
  --verbose            print every move
  --help               usage text
```

Colours alternate between games, and both colours of one opening pair use the
same opening (opening index = `game / 2`), so the result is not biased by the
openings. Games start from a short random opening (`--open-plies`, default 6
plies) or from FENs you supply:

```bash
printf 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n' > openings.fen
./arena --openings openings.fen --games 4
```

## Examples

```bash
# default match, 20 games, depth 8, quiescence on
./arena

# does quiescence search help? (same openings for both halves of the match)
./arena --qs both --games 40 --depth 8

# fast smoke test with a node budget instead of a depth budget
./arena --games 4 --depth 64 --nodes 20000

# watch one game move by move, write the PGN
./arena --games 2 --depth 6 --verbose --pgn match.pgn

# weaker, more "human" policy opponent
./arena --games 20 --policy-temp 0.8
```

## Tips

* A raw policy net (no search) is much weaker than even a depth-1 search. For
  closer, more informative matches use `--depth 1`..`3`, a node budget, or a
  small `--movetime`, and/or give the policy some randomness with
  `--policy-temp`.
* `--qs both` is a comparison tool: with few games the difference is pure noise
  (look at the printed Elo interval). Use at least a few hundred games before
  drawing conclusions.
* The NNUE and policy nets must match the hidden sizes the binary was compiled
  for (`KOCIOLEK_POLICY_HL`, `POLICY_SMALL_HL`); a size mismatch is reported as
  a load error.
* `--verbose` shows every move, including the random opening plies.

## Sample output

```
 match: Kociolek NNUE (search)  vs  policy net (no search)
 quiescence search : ON
 nnue player       : depth 8, search-policy on
 policy player     : big net, argmax
 games             : 20
=====================================================================
game   1/20  NNUE=w Policy=b  1-0  (checkmate)  plies 61  0.4s  nnue_nodes 812233
...
 NNUE wins 16   policy wins 3   draws 1   (20 games)
 NNUE score 82.50%   Elo +266  [12, 520]
```
