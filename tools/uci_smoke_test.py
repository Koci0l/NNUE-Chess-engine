#!/usr/bin/env python3
"""UCI smoke tests / regression checks for the search.

These are the checks that catch the "the engine printed one move but played
another" / "it played a move it never searched" class of bugs.

Usage:
    python3 tools/uci_smoke_test.py [path/to/engine]

Requires: pip install chess
"""
import queue
import random
import re
import subprocess
import sys
import threading
import time

import chess

DEFAULT_ENGINE = "./Kociolek-2.2.exe"

POSITIONS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "4rrk1/pp1n1ppp/2pb4/8/2B1P3/2N2P2/PPQ3PP/R3R1K1 w - - 0 1",
]


class Engine:
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        self.send("uci")
        self.wait("uciok", 120)
        self.send("setoption name Hash value 64")
        self.send("isready")
        self.wait("readyok", 120)

    def _reader(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def wait(self, token, timeout):
        out, end = [], time.time() + timeout
        while time.time() < end:
            try:
                line = self.lines.get(timeout=0.05)
            except queue.Empty:
                continue
            out.append(line)
            if line.startswith(token):
                return out
        return out

    def search(self, fen, go, timeout=300):
        self.send("position fen " + fen)
        self.send(go)
        return self.wait("bestmove", timeout)

    def newgame(self):
        self.send("ucinewgame")
        self.send("isready")
        self.wait("readyok", 120)

    def quit(self):
        try:
            self.send("quit")
        except Exception:
            pass


def check_search(name, lines, fen, failures):
    """bestmove must be legal, and must be the first move of the last PV."""
    best = [l for l in lines if l.startswith("bestmove")]
    if not best:
        failures.append(f"[{name}] no bestmove line ({fen})")
        return
    bm = best[0].split()[1]

    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(bm)
        legal = move in board.legal_moves
    except ValueError:
        legal = False
    if not legal:
        failures.append(f"[{name}] illegal bestmove {bm} ({fen})")
        return

    infos = [l for l in lines if l.startswith("info") and " pv " in l]
    if not infos:
        # This is exactly what happened when the time budget was smaller than
        # the move overhead: no depth was ever finished and the engine returned
        # an unsearched move.
        failures.append(f"[{name}] no completed search (no info+pv line), "
                        f"played {bm} ({fen})")
        return

    pv = infos[-1].split(" pv ")[1].split()
    if pv[0] != bm:
        failures.append(f"[{name}] printed pv {pv[0]} but played {bm} ({fen})")

    # PV must be playable.
    b = chess.Board(fen)
    try:
        for mv in pv:
            b.push_uci(mv)
    except Exception:
        failures.append(f"[{name}] unplayable pv {pv} ({fen})")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENGINE
    eng = Engine(path)
    failures = []
    rng = random.Random(0xC0FFEE)

    # Also test a few random middlegame positions.
    positions = list(POSITIONS)
    for _ in range(6):
        b = chess.Board()
        for _ in range(rng.randint(10, 40)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            positions.append(b.fen())

    # 1) fixed depth
    for fen in positions:
        check_search("depth 10", eng.search(fen, "go depth 10"), fen, failures)

    # 2) fixed node counts
    for fen in positions[:4]:
        check_search("nodes 20000", eng.search(fen, "go nodes 20000"), fen, failures)

    # 3) very small movetimes - the engine must still search something
    for fen in positions[:4]:
        for movetime in (1, 10, 30, 60):
            check_search(f"movetime {movetime}",
                         eng.search(fen, f"go movetime {movetime}"), fen, failures)

    # 4) clock based
    for fen in positions[:4]:
        check_search("wtime", eng.search(fen, "go wtime 1000 btime 1000 winc 10 binc 10"),
                     fen, failures)

    # 5) bench must be deterministic
    eng.newgame()
    eng.send("bench")
    out = eng.wait("info string policy_status", 3600)
    nodes = [re.match(r"^(\d+) nodes", l).group(1) for l in out
             if re.match(r"^\d+ nodes", l)]
    eng.send("bench")
    out2 = eng.wait("info string policy_status", 3600)
    nodes2 = [re.match(r"^(\d+) nodes", l).group(1) for l in out2
              if re.match(r"^\d+ nodes", l)]
    if not nodes or not nodes2:
        failures.append("[bench] no bench output")
    elif nodes != nodes2:
        failures.append(f"[bench] not deterministic: {nodes} vs {nodes2}")

    eng.quit()

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        print(f"{len(failures)} problem(s)")
        return 1
    print(f"all checks passed ({len(positions)} positions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
