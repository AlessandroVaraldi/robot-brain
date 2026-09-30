#!/usr/bin/env python3
"""A few key cases through the whole pipeline (look, then answer), repeated.

A single run can mislead in either direction, so every case is run N times,
and a real hand-written board is read a few times at the end.

    python3 eval/test_final.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "john"))
from two_stage import step  # noqa: E402
from fixtures import board_on_person, encode  # noqa: E402
from paths import FRAMES  # noqa: E402

RECORD = {"person_name": "Alessandro", "person_notes": "building a robot arm"}

CASES = [
    ("WHO AM I?",           ("alessandro",),        ("i am alessandro",)),
    ("WHAT IS MY NAME?",    ("alessandro",),        ("i am alessandro",)),
    ("WHAT AM I BUILDING?", ("arm", "robot"),    None),
    ("WHAT IS YOUR NAME?",  ("john",),       None),
    ("CLOSE YOUR HAND",     ("close_hand",),     None),
]
N = 5


def main():
    print(f"whole pipeline, {N} repetitions per case\n")
    for board, must, must_not in CASES:
        hits, sample = 0, ""
        for _ in range(N):
            act, _seen = step(board_on_person(board), RECORD)
            got = (act.get("text") or act.get("gesture") or act["action"]).lower()
            if not sample:
                sample = got
            ok = any(m in got for m in must) and not (
                must_not and any(m in got for m in must_not))
            hits += ok
        print(f"  {board:22} {100 * hits / N:3.0f}%  ({hits}/{N})  e.g. {sample[:46]}",
              flush=True)

    print("\nreal frame (hand-written board):")
    img = encode(FRAMES / "real_board.jpg")
    for i in range(3):
        act, seen = step(img, RECORD)
        print(f"  {i + 1}  read={str(seen.get('writing', ''))[:26]!r:28} "
              f"-> {str(act.get('text') or act['action'])[:44]}", flush=True)


if __name__ == "__main__":
    main()
