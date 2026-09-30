#!/usr/bin/env python3
"""The suite's answer cases for some categories only, P0 wording, N repetitions.
For quickly re-measuring the part a change touches; the full check is robust.py.

    python3 eval/quick_answers.py                 # default categories
    python3 eval/quick_answers.py others unknowable
"""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import two_stage as T  # noqa: E402
import suite  # noqa: E402

N = 4
CATS = set(sys.argv[1:]) or {"statement_person", "person", "mixed", "memory"}


def main():
    T.warmup()
    tot = ok_all = 0
    for cat, ctx, board, must, must_not, must_all in suite.A:
        if cat not in CATS:
            continue
        rec, mem = suite.CTX[ctx]
        outs = [T.answer({"writing": board, "engaged": True, "scene": suite.SCENE}, dict(rec), mem, suite.SELF)
                for _ in range(N)]
        ok = sum(not suite.check(o, must, must_not, must_all) for o in outs)
        tot += N; ok_all += ok
        top = Counter(outs).most_common(1)[0][0]
        print(f"  {ok}/{N}  [{cat}/{ctx}] {board[:36]:38} {top[:60]}")
    print(f"\ntotal {ok_all}/{tot}")


if __name__ == "__main__":
    main()
