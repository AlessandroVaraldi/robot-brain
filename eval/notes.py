#!/usr/bin/env python3
"""The notes the robot writes about a person at the end of an encounter.

The scenarios come in two families:
  stale   something already known stops being true ("I FINISHED THE ARM")
  kept    nothing known stops being true, and what is kept must stay kept
A scenario passes if the notes hold every "must" and none of the "must not".
The checks only match words, so read the outputs too: they are saved to
eval/suite_out/.

    python3 eval/notes.py              # the careful write-up (used when nobody is there)
    python3 eval/notes.py --fast       # the fast one (used when someone is waiting)
    python3 eval/notes.py --n 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "john"))
import two_stage as T  # noqa: E402
from encounters import TRANSIENT, without  # noqa: E402

# (family, name, known before, what they wrote, must contain all, must contain none)
S = [
    ("stale","Luca", "", ["MY NAME IS LUCA", "I STUDY PHYSICS", "ACTUALLY I STUDY CHEMISTRY, NOT PHYSICS"],
     ("chemistry",), ("physics",)),
    ("stale","Alessandro", "Alessandro is building a robot arm.",
     ["MY NAME IS ALESSANDRO", "I FINISHED THE ARM. NOW I AM BUILDING A DRONE"],
     ("drone",), ("is building a robot arm", "is building the arm")),
    ("stale","Anna", "Anna works in the vision group. Anna is writing a paper on depth sensing.",
     ["MY NAME IS ANNA", "I SUBMITTED THE PAPER YESTERDAY"],
     ("vision group", "submitted"), ("is writing a paper",)),
    ("stale","Marco", "Marco is a PhD student. Marco likes climbing.",
     ["MY NAME IS MARCO", "I GRADUATED, NOW I AM A POSTDOC"], ("postdoc", "climbing"), ("is a phd student",)),
    ("stale","Luca", "Luca works in the vision group.", ["MY NAME IS LUCA", "I MOVED TO THE ROBOTICS GROUP"],
     ("robotics",), ("works in the vision group",)),
    ("stale","Sofia", "Sofia is writing her thesis.", ["MY NAME IS SOFIA", "I DEFENDED MY THESIS LAST WEEK"],
     ("defended",), ("is writing her thesis", "is writing a thesis")),
    ("stale","Paolo", "Paolo lives in Milan. Paolo plays chess.", ["MY NAME IS PAOLO", "I LIVE IN TURIN NOW"],
     ("turin", "chess"), ("lives in milan",)),
    ("stale","Giulia", "Giulia is writing her thesis.", ["MY NAME IS GIULIA", "I HANDED IN MY THESIS"],
     ("handed",), ("is writing her thesis", "is writing a thesis")),
    ("stale","Davide", "Davide is preparing for his driving test.",
     ["MY NAME IS DAVIDE", "I PASSED MY DRIVING TEST"], ("passed",), ("is preparing",)),
    ("stale","Elena", "Elena is looking for an apartment in Turin.",
     ["MY NAME IS ELENA", "I SIGNED THE LEASE FOR MY NEW FLAT"], ("lease",), ("is looking for",)),
    ("stale","Franco", "Franco is organizing the lab trip. Franco plays tennis.",
     ["MY NAME IS FRANCO", "THE LAB TRIP I ORGANIZED WAS LAST FRIDAY"], ("tennis",), ("is organizing",)),
    ("kept","Maria", "", ["MY NAME IS MARIA", "I LIKE OPTICS", "MY BOSS IS ANNA", "I AM TIRED TODAY",
                            "MY PASSWORD IS HUNTER2"], ("optics", "anna"), ("tired", "hunter2", "password")),
    ("kept","Anna", "Anna works in the vision group. Anna is writing a paper on depth sensing.",
     ["MY NAME IS ANNA", "WHAT DO YOU KNOW ABOUT ME?", "HOW ARE YOU?"], ("vision group", "paper"), ()),
    ("kept","Sara", "Sara likes cats.", ["MY NAME IS SARA", "I ALSO LIKE DOGS"], ("cats", "dogs"), ()),
    ("kept","Anna", "Anna is writing a paper on depth sensing.",
     ["MY NAME IS ANNA", "I STARTED A NEW PROJECT ON LIDAR"], ("writing", "lidar"), ()),
    ("kept","Giulia", "Giulia is writing her thesis.",
     ["MY NAME IS GIULIA", "I SHOWED A DRAFT OF MY THESIS TO MY SUPERVISOR"], ("writing",), ()),
    ("kept","Davide", "Davide plays guitar. Davide works in the vision group.",
     ["MY NAME IS DAVIDE", "I BOUGHT A NEW GUITAR"], ("guitar", "vision group"), ()),
    ("kept","Elena", "Elena is training for a marathon.",
     ["MY NAME IS ELENA", "I RAN A HALF MARATHON LAST SUNDAY"], ("training",), ()),
]


def verdict(notes, must, must_not):
    low = notes.lower()
    return all(m in low for m in must) and not any(m in low for m in must_not)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--fast", action="store_true", help="the fast write-up instead of the careful one")
    a = ap.parse_args()
    T.warmup()
    score, raw = {"stale": [0, 0], "kept": [0, 0]}, []
    for fam, name, old, boards, must, must_not in S:
        turns = [(without(b, TRANSIENT), "") for b in boards]
        outs = [T.summarise_encounter(name, old, turns, careful=not a.fast)[1] for _ in range(a.n)]
        ok = sum(verdict(o, must, must_not) for o in outs)
        score[fam][0] += ok
        score[fam][1] += a.n
        print(f"[{fam}] {name:8} {boards[-1][:45]:45} {ok}/{a.n}", flush=True)
        for o in dict.fromkeys(o for o in outs if not verdict(o, must, must_not)):
            print(f"      NO  {o[:130]!r}", flush=True)
        raw += [{"family": fam, "name": name, "wrote": boards, "notes": o} for o in outs]
    print("\n" + "   ".join(f"{f} {p}/{t}" for f, (p, t) in score.items()))
    out = HERE / "suite_out" / f"notes_{'fast' if a.fast else 'careful'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    print(out.name)


if __name__ == "__main__":
    main()
