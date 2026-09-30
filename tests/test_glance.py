#!/usr/bin/env python3
"""glance.py: which writing asks for a look, and the facts it gives.  The
vision model and the face detector are stand-ins: no models, no OpenCV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import glance  # noqa: E402
from tricks import said  # noqa: E402

COUNT = [
    ("HOW MANY OF US ARE THERE?", True),
    ("HOW MANY PEOPLE CAN YOU SEE?", True),
    ("HOW MANY PEOPLE ARE HERE?", True),
    ("HOW MANY FACES DO YOU SEE", True),
    ("HOW MANY PEOPLE DID YOU MEET TODAY?", False),     # memory, not the picture
    ("HOW MANY PEOPLE DO YOU KNOW?", False),
    ("HOW MANY HANDS DO YOU HAVE?", False),
    ("WHAT IS 2+2?", False),
]
DRAW = [
    ("GUESS WHAT I DREW", True), ("GUESS WHAT I DREW!", True), ("WHAT DID I DRAW?", True),
    ("WHAT AM I DRAWING?", True), ("WHAT IS THIS DRAWING?", True), ("GUESS THE DRAWING", True),
    ("I LIKE DRAWING", False), ("WHAT IS YOUR NAME?", False),
]


def main():
    failed = 0

    def check(ok, label):
        nonlocal failed
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")

    for writing, want in COUNT:
        check(glance.asks_count(writing) == want, f"count?  {writing!r} -> {want}")
    for writing, want in DRAW:
        check(bool(glance.DRAW_Q.search(writing)) == want, f"draw?   {writing!r} -> {want}")

    three = glance.count_fact(3)
    check(said(three, "I see three of you.") and said(three, "3 people, I think."),
          "count 3: '3' or 'three' both count as said")
    check(not said(three, "Two of you."), "count 3: 'two' does not")
    check(glance.count_fact(1).text.endswith("right now: 1."), "count 1")
    check(glance.count_fact(0).say == "", "count 0: nothing to insist on")

    check("an umbrella" in glance.drawing_fact("Umbrella", "mushroom").text, "an umbrella")
    check("a house - or perhaps a barn" in glance.drawing_fact("house", "barn").text, "a house or a barn")
    check("perhaps" not in glance.drawing_fact("sun", "Sun").text, "no second guess equal to the first")
    check("cannot make out" in glance.drawing_fact("", "").text, "nothing drawn")

    asked = []
    eye = lambda prompt, img: asked.append(prompt) or {"guess": "sun", "other": "flower"}  # noqa: E731
    f = glance.glance("GUESS WHAT I DREW", "IMG", eye)
    check(f is not None and f.say == "sun" and asked == [glance.DRAW_PROMPT], "drawing: one question to the eye")
    glance.count_faces = lambda img: 2
    f = glance.glance("HOW MANY OF US ARE THERE?", "IMG", eye)
    check(f is not None and f.text.endswith(": 2.") and len(asked) == 1, "count: the detector, not the eye")
    glance.count_faces = lambda img: None
    check(glance.glance("HOW MANY OF US ARE THERE?", "IMG", eye) is None, "count: off without a detector")
    check(glance.glance("GUESS WHAT I DREW", None, eye) is None, "no picture: nothing")
    check(glance.glance("WHAT IS YOUR NAME?", "IMG", eye) is None and len(asked) == 1,
          "other writing: the eye is not asked")

    total = len(COUNT) + len(DRAW) + 13
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
