#!/usr/bin/env python3
"""Blindfold chess with the real eye and the real engine, robot not involved.

1. reading: boards with moves and commands through the eye - is the writing
   copied exactly, capitals included, and does it give the move meant?
2. a game: a random player (fixed seed) writes its moves on boards, half in
   from-to squares, half in standard notation, against John through step().
   A misread that is still a legal move is taken back with UNDO LAST MOVE and
   written again, as a person would.

    python3 eval/chess_eval.py
"""

import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "john"))
import chess  # noqa: E402
import chess_table as ct  # noqa: E402
import two_stage as T  # noqa: E402
from fixtures import board_on_person  # noqa: E402

READING = ["e2 e4", "E2 E4", "g1 f3", "Nf3", "Bc4", "bxc5", "O-O", "O-O-O", "Qxf7#",
           "e7e8=Q", "Kd2", "UNDO LAST MOVE", "I RESIGN", "DRAW?"]


def reading():
    print("reading (the eye's transcription)")
    exact = same_letters = 0
    for text in READING:
        got = str(T.look(board_on_person(text)).get("writing", ""))
        exact += got.replace(" ", "") == text.replace(" ", "")
        same_letters += got.replace(" ", "").lower() == text.replace(" ", "").lower()
        print(f"  {text:16} -> {got!r}")
    print(f"  exact {exact}/{len(READING)}, same ignoring capitals {same_letters}/{len(READING)}")


def game(seed=7, max_moves=40):
    print("\na game against a random player")
    rng = random.Random(seed)
    T.TABLE = ct.ChessTable(clock=time.time, rng=rng)
    record = {"person_name": "", "person_notes": "", "pending": ""}

    def show(text):
        t = time.time()
        act, seen = T.step(board_on_person(text), record)
        return act, seen, time.time() - t

    act, _, _ = show("LET'S PLAY CHESS")
    print(f"  LET'S PLAY CHESS -> {act.get('text')!r}")
    act, _, _ = show("WHITE")
    print(f"  WHITE -> {act.get('text')!r}")
    times, misreads, undos, turns = [], 0, 0, 0
    while T.TABLE.state == "play" and turns < max_moves:
        board = T.TABLE.board
        move = rng.choice(sorted(board.legal_moves, key=lambda m: m.uci()))
        written = move.uci()[:2] + " " + move.uci()[2:] if turns % 2 == 0 else board.san(move)
        n = len(board.move_stack)
        act, seen, dt = show(written)
        turns += 1
        played = T.TABLE.board.move_stack[n] if len(T.TABLE.board.move_stack) > n else None
        if played != move:
            misreads += 1
            print(f"  wrote {written!r}, eye read {seen.get('writing')!r} -> {act.get('text', '')[:70]!r}")
            if played is not None:                       # a legal move, but not ours
                show("UNDO LAST MOVE")
                undos += 1
            if T.TABLE.state == "quit?":
                show("NO")
            continue
        times.append(dt)
        if turns <= 3 or T.TABLE.state != "play":
            print(f"  {written:8} -> {act.get('text')!r}  ({dt:.1f}s)")
    print(f"  moves written {turns}, misread {misreads} (taken back {undos}), state {T.TABLE.state}")
    if times:
        print(f"  seconds per turn: median {statistics.median(times):.1f}, max {max(times):.1f}")
    if T.TABLE.state == "rematch?":
        act, _, _ = show("NO")
        print(f"  NO -> {act.get('text')!r}")
    T.TABLE.reset()


if __name__ == "__main__":
    T.warmup()
    reading()
    game()
