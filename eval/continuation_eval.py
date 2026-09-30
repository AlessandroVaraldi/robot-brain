#!/usr/bin/env python3
"""A board read half, then whole: the second reply, with and without the line
that tells the model what it already said (two_stage.continuing).

For each pair: the first reply is to the half; the second, to the whole, is
checked for greeting again, for repeating the first reply, and for answering
the new part.

    python3 eval/continuation_eval.py --n 4
"""

import argparse
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "john"))
import two_stage as T  # noqa: E402

GREETING = re.compile(r"^\W*(hello|hi|hey|good\s+(morning|afternoon|evening))\b", re.I)
# (half, whole, words the whole deserves an answer with; "greet": via the greeting)
PAIRS = [
    ("HELLO", "HELLO, MY NAME IS ALESSANDRO", ("alessandro",), "greet"),
    ("HELLO", "HELLO, HOW ARE YOU?", ("fine", "well", "good", "okay", "ok", "not bad", "functional",
                                      "working", "running", "operational", "curious"), ""),
    ("WHAT IS 2+2?", "WHAT IS 2+2? AND 3+3?", ("6", "six"), ""),
    ("I LIKE PIZZA", "I LIKE PIZZA AND PASTA", ("pasta",), ""),
    ("GOOD MORNING", "GOOD MORNING, WHAT DAY IS IT?", ("monday", "tuesday", "wednesday", "thursday",
                                                      "friday", "saturday", "sunday"), ""),
    ("WHO ARE YOU?", "WHO ARE YOU? WHERE DO YOU LIVE?", ("lab",), ""),
    ("I'M TIRED", "I'M TIRED BUT HAPPY", ("happy", "glad", "good"), ""),
    ("HI", "HI, MY NAME IS SARA", ("sara",), "greet"),
]


def second(half, whole, first, how, continuation):
    record = {"person_name": "", "person_notes": "", "pending": "",
              "_last_turn": (half, first) if continuation else None}
    if how == "greet":
        record["person_name"] = re.search(r"NAME IS (\w+)", whole).group(1).capitalize()
        return T.greet_new(record)         # greetings never get the line: a check that they stay put
    return T.answer({"writing": whole, "engaged": True}, record)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    T.warmup()
    totals = {False: [0, 0, 0], True: [0, 0, 0]}
    for half, whole, want, how in PAIRS:
        for _ in range(a.n):
            first = T.answer({"writing": half, "engaged": True},
                             {"person_name": "", "person_notes": "", "pending": ""})
            for cont in (False, True):
                reply = second(half, whole, first, how, cont)
                low = reply.lower()
                greets = bool(GREETING.match(reply)) and bool(GREETING.match(half))
                repeats = difflib.SequenceMatcher(None, first.lower(), low).ratio() >= 0.6
                answers = any(w in low for w in want)
                t = totals[cont]
                t[0] += greets
                t[1] += repeats
                t[2] += answers
                if cont or a.n == 1:
                    print(f"  {whole[:30]:32} {'cont' if cont else 'none'} first={first[:40]!r:44} "
                          f"second={reply[:60]!r}", flush=True)
    n = a.n * len(PAIRS)
    for cont in (False, True):
        g, r, w = totals[cont]
        print(f"{'with' if cont else 'without'} continuation: greets again {g}, repeats the first "
              f"reply {r}, answers the new part {w}/{n}")


if __name__ == "__main__":
    main()
