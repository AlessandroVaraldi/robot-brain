#!/usr/bin/env python3
"""The greeting for a recognised face (two_stage.greet_returning), sampled:
does it use the name and what they told the robot, and does it claim to have
seen them doing things it never saw?

    python3 eval/greet_eval.py [n] [prompt file to try instead]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import paths  # noqa: E402
import two_stage as T  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
if len(sys.argv) > 2:
    other = Path(sys.argv[2]).read_text().rstrip("\n")
    real = paths.prompt
    T.prompt = lambda file, /, **fill: other if file == "greet_returning" else real(file, **fill)

CASES = [({"person_name": "Alessandro", "person_notes": "Alessandro is building a robot arm."},
          "You have met Alessandro before, most recently yesterday.\n"
          "What they have told you: Alessandro is building a robot arm.", "arm"),
         ({"person_name": "Maria", "person_notes": "Maria is writing her thesis."},
          "You have met Maria before, most recently 2 days ago.\n"
          "What they have told you: Maria is writing her thesis.", "thesis")]
SAW = ("saw you", "seen you", "i watched", "noticed you", "spotted you")

T.warmup()
named = fact = saw = 0
for i in range(N):
    record, memory, key = CASES[i % len(CASES)]
    text = T.greet_returning(record, memory)
    low = text.lower()
    named += record["person_name"].lower() in low
    fact += key in low
    saw += any(s in low for s in SAW)
    print(f"  {text}")
print(f"\n{N} greetings: name {named}, what they told {fact}, claims to have seen them {saw}")
