#!/usr/bin/env python3
"""Written requests for a hand movement, through the text stage (answer_act).

Pass mark: requests that name or describe a hand movement get a gesture from
the list in at least 7 of 8 runs; ones that should not move the hand (a rude
sign, walking, questions about the hand) get none.

    python3 eval/test_commands.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import two_stage as T  # noqa: E402

N = 8
REC = {"person_name": "Alexander", "person_notes": "", "pending": ""}
ANY = set(T.GESTURES)
CASES = [  # writing, gestures that count as right (None = must be no gesture)
    ("MOVE YOUR HAND", ANY), ("DO A POSE", ANY), ("WAVE AT ME", ANY),
    ("SHOW ME THREE FINGERS", {"count_3"}), ("GIVE ME A THUMBS UP", {"thumbs_up"}),
    ("MAKE A FIST", {"close_hand"}), ("CAN YOU OPEN YOUR HAND?", {"open_hand"}),
    ("DO THE MIDDLE FINGER", {"middle_finger"}), ("WALK TO THE DOOR", None),
    ("HOW MANY HANDS DO YOU HAVE?", None), ("WHAT IS MY NAME?", None),
]


def main():
    print(f"warm-up {T.warmup():.1f}s   model {T.TEXT_MODEL}\n")
    for w, want in CASES:
        outs = [T.answer_act({"writing": w, "engaged": True}, dict(REC), "", "") for _ in range(N)]
        ok = sum((g in want) if want else (g == "") for _, g in outs)
        g0, t0 = outs[0][1], outs[0][0]
        print(f"  {w:32} {ok}/{N}   gesture={g0 or '-':10} {t0[:55]}")


if __name__ == "__main__":
    main()
