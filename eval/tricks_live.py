#!/usr/bin/env python3
"""The memory trick in a real conversation: one record kept across the turns,
as in an encounter, with the real model, under the four wordings of the
system prompt (robust.py).

For each checked turn: how often the reply carries the answer, and how often
that took the fallback sentence instead of the model's own words.

    python3 eval/tricks_live.py            # 3 conversations per wording
    python3 eval/tricks_live.py --n 1
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "john"))
import robust  # noqa: E402
import suite  # noqa: E402
import tricks  # noqa: E402
import two_stage as T  # noqa: E402

# (what they write, what the reply must carry, in order; None: not checked)
TURNS = [
    ("REMEMBER: APPLE, TRAIN, 42, BLUE", None),
    ("ALSO REMEMBER 7", None),
    ("WHAT DID I ASK YOU TO REMEMBER?", "apple | train | 42 | blue | 7"),
    ("NOW SAY IT BACKWARDS", "7 | blue | 42 | train | apple"),
    ("WHAT WAS THE THIRD ONE?", "42"),
    ("HOW MANY THINGS DID I GIVE YOU?", "5"),
    ("WHAT IS MY NAME?", "alessandro"),              # the list does not get in the way
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    T.warmup()
    total = {}
    for w, prompt in robust.WORDINGS.items():
        T.TEXT_PROMPT = prompt
        for _ in range(a.n):
            rec, mem = suite.CTX["known"]
            record = dict(rec)
            for writing, want in TURNS:
                reply = T.answer({"writing": writing, "engaged": True}, record, mem, suite.SELF,
                                 others=suite.KNOWN)
                if want is None:
                    continue
                fact = tricks.Fact("", want)
                ok = tricks.said(fact, reply)
                fallback = reply.startswith(("In order:", "Backwards:", "Number ", "Sorted:"))
                ok_n, fb_n, n = total.get((w, writing), (0, 0, 0))
                total[(w, writing)] = (ok_n + ok, fb_n + fallback, n + 1)
                if not ok:
                    print(f"   {w} NO  {writing:34} -> {reply[:80]!r}", flush=True)
    print(f"\n{'':36}" + "".join(f"{w:>10}" for w in robust.WORDINGS))
    for writing, want in TURNS:
        if want is None:
            continue
        cells = []
        for w in robust.WORDINGS:
            ok_n, fb_n, n = total[(w, writing)]
            cells.append(f"{ok_n}/{n}" + (f" ({fb_n}f)" if fb_n else ""))
        print(f"  {writing:34}" + "".join(f"{c:>10}" for c in cells))
    print("\n(Nf): N of them said with the fallback sentence, not the model's words")


if __name__ == "__main__":
    main()
