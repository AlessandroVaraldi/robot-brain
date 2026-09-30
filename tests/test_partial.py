#!/usr/bin/env python3
"""Boards read half, then whole: what is held back, and the line that tells the
reply what it already said.  The readings are from live sessions.  No models."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import two_stage as T  # noqa: E402
from encounters import EncounterTracker, Mind  # noqa: E402
from memory_store import MemoryStore  # noqa: E402

RESULTS = []


def check(ok, label):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


HELD = ["...", "?", "…", "SUCK YOUR", "Hi I'm", "MOVE YOUR", "How do", "Who do", "What have been",
        "IM", "Are y", "I AM THE", "Hello,", "HELLO, MY NAME IS:", "FIRST QUESTION -"]
PASSED = ["HELLO", "WHO BUILT you?", "4+2", "I'M HAPPY", "Thank you.", "WHAT IS THIS?", "DO A POSE",
          "I AM YOUR CREATOR", "MY NAME IS ALESSANDRO", "HOW OLD AM I?", "SAY HELLO TO ME",
          "O-O", "0-0", "e2 e4", "Nf3", "Qxf7#", "I'M 28.\nHOW OLD AM I?", "WHO ARE YOU?"]


def main():
    for w in HELD:
        check(T.unfinished(w), f"held:   {w!r}")
    for w in PASSED:
        check(not T.unfinished(w), f"passed: {w!r}")

    rec = {"_last_turn": ("HELLO", "Hello, are you new here?")}
    line = T.continuing(rec, "HELLO, MY NAME IS ALESSANDRO")
    check('said only "HELLO"' in line and 'also says "MY NAME IS ALESSANDRO"' in line
          and "new here" not in line, "continuation: what was answered and what is new, not the reply")
    line = T.continuing({"_last_turn": ("WHAT IS 2+2?", "4.")}, "WHAT IS 2+2? AND 3+3?")
    check('also says "AND 3+3?"' in line, f"the added part, as written: {line!r}")
    check(T.continuing(rec, "HELLO") == "", "the same board: no continuation")
    check(T.continuing(rec, "WHAT IS YOUR NAME?") == "", "another board: no continuation")
    check(T.continuing({"_last_turn": None}, "HELLO, MY NAME IS ALESSANDRO") == "", "nothing said lately")
    T.CONTINUATION = False
    check(T.continuing(rec, "HELLO, MY NAME IS ALESSANDRO") == "", "switched off")
    T.CONTINUATION = True

    # step(): a half-read board is left alone - not marked as seen, record untouched
    T.look = lambda img: {"writing": "SUCK YOUR", "engaged": True, "route": "answer"}
    asked = []
    record = {"person_name": "", "person_notes": "", "pending": ""}
    act, seen = T.step("img", record, keep_record=True, is_new=lambda s: asked.append(s) or True)
    check(act == {"action": "none", "unfinished": True} and not asked and "_last_turn" not in record,
          "step: held, the repeat check not consulted, the record untouched")

    # Mind: the last turn is noted for 15 s; a held board adds no turn
    now = [0.0]
    replies = iter(["Hello, are you new here?", None, "Nice to meet you."])
    seen_records = []

    def decide(image, record, memory, is_new=None):
        seen_records.append(dict(record))
        r = next(replies)
        if r is None:
            return {"action": "none", "unfinished": True}, {"writing": "...", "engaged": True}
        return {"action": "speak", "text": r}, {"writing": image, "engaged": True}

    mind = Mind(MemoryStore(Path(tempfile.mkdtemp()) / "m.sqlite"), EncounterTracker(timeout_s=30),
                decide, lambda n, o, t: ("", ""), lambda s, r, m: "", clock=lambda: now[0])
    for img, dt in (("HELLO", 1), ("...", 2), ("HELLO, HOW ARE YOU?", 2)):
        now[0] += dt
        mind.on_frame(True, img)
    enc = mind.tracker.active
    check(seen_records[0].get("_last_turn") is None, "first board: nothing said before")
    check(seen_records[2].get("_last_turn") == ("HELLO", "Hello, are you new here?"),
          "a moment later: what was said, and to what")
    check([w for w, _ in enc.turns] == ["HELLO", "HELLO, HOW ARE YOU?"], "the held reading is not a turn")
    now[0] += 20
    replies = iter(["later"])
    mind.on_frame(True, "SOMETHING ELSE")
    check(seen_records[-1].get("_last_turn") is None, "after 15 s: not a continuation any more")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
