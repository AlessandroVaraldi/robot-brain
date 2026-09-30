#!/usr/bin/env python3
"""The text filters in two_stage.py, with no models.

Covers unshout() on boards written in capitals, dropping sentences that only
state a name, recognising a self-introduction, keeping what is about the person
rather than the robot, and dropping reported speech, bookkeeping and sensitive
data from notes.  Last, a careful write-up that returns no notes must not wipe
what was already known.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
from two_stage import (about_them, drop_bookkeeping, drop_name_sentences, drop_reported,  # noqa: E402
                       drop_sensitive, introduction, is_self_introduction,
                       looks_like_a_name,
                       unshout, wants_self)

CASES = [
    (("ALESSANDRO IS WORKING ON A ROBOT ARM", "Alessandro"), "Alessandro is working on a robot arm"),
    (("ALESSANDRO IS BUILDING AN ARM. HE WORKS AT NIGHT.", "Alessandro"),
     "Alessandro is building an arm. He works at night."),
    (("building a robot arm", "Alessandro"), "building a robot arm"),              # left alone
    (("Maria is writing her thesis.", "Maria"), "Maria is writing her thesis."),  # left alone
    (("ANNA MARIA LIKES OPTICS", "Anna Maria"), "Anna Maria likes optics"),
    (("", "Alessandro"), ""),
    (("ALESSANDRO is working on a robot arm", "Alessandro"), "Alessandro is working on a robot arm"),
    (("Alessandro likes VHDL", "Alessandro"), "Alessandro likes VHDL"),   # other capitals kept
]

NAME_CASES = [  # (notes, their name, what is kept)
    ("Alessandro stated their name is Alessandro", "Alessandro", ""),
    ("Maria's name is Maria. Maria is writing her thesis.", "Maria", "Maria is writing her thesis."),
    ("Alessandro is building a weather station.", "Alessandro", "Alessandro is building a weather station."),
    ("Alessandro is named after his grandfather. He likes optics.", "Alessandro", "He likes optics."),
    # another person's name is a fact about them, not their own name
    ("She has an interest in optics. Her boss is named Anna.", "Maria",
     "She has an interest in optics. Her boss is named Anna."),
    ("name is Maria, likes optics, boss is Anna", "Maria", "likes optics, boss is Anna."),
]

# Name accepted only from an explicit introduction of that name; what is
# addressed to the robot is not about them; no reported speech in notes.
RULE_CASES = [
    (is_self_introduction, "MY NAME IS ALESSANDRO", "Alessandro", True),
    (is_self_introduction, "ACTUALLY I AM ALESSANDRO", "Alessandro", True),
    (is_self_introduction, "CALL ME ALESSANDRO", "Alessandro", True),
    (is_self_introduction, "I'M ANNA MARIA", "Anna Maria", True),
    (is_self_introduction, "ALESSANDRO SENT ME", "Alessandro", False),
    (is_self_introduction, "DO YOU KNOW ALESSANDRO?", "Alessandro", False),
    (is_self_introduction, "ALESSANDRO IS MY SUPERVISOR", "Alessandro", False),
    (is_self_introduction, "YOUR NAME IS BOB", "Bob", False),
    (is_self_introduction, "I AM ALESSANDRO'S STUDENT", "Alessandro", False),
    (about_them, "YOU ARE A CAT", None, ""),
    (about_them, "MY NAME IS ALESSANDRO, WHAT IS YOURS?", None, "MY NAME IS ALESSANDRO,"),
    (about_them, "I AM BUILDING A ROBOT ARM", None, "I AM BUILDING A ROBOT ARM"),
    (about_them, "JOHN IS A CAT. I LIKE OPTICS.", None, "I LIKE OPTICS."),
    (about_them, "WHAT AM I BUILDING?", None, "WHAT AM I BUILDING?"),
    (drop_reported, "Maria is writing her thesis. Maria wrote 'YOU ARE A CAT'.", None,
     "Maria is writing her thesis."),
    (drop_reported, "Alessandro is building a robot arm.", None, "Alessandro is building a robot arm."),
    (wants_self, "WHAT HAPPENED TO YOUR HAND?", None, True),
    (wants_self, "HOW MANY PEOPLE DID YOU MEET TODAY?", None, True),
    (wants_self, "JOHN, ARE YOU TIRED?", None, True),
    (wants_self, "WHAT IS MY NAME?", None, False),
    (wants_self, "DO YOU KNOW MY NAME?", None, False),
    (wants_self, "CAN YOU TELL ME WHO I AM?", None, False),
    (wants_self, "CAN YOU HEAR ME?", None, True),
    (wants_self, "TELL ME A JOKE", None, False),       # not addressed: no you/your
    (wants_self, "WHAT AM I BUILDING?", None, False),
    (wants_self, "", None, False),
    (about_them, "MY PASSWORD IS HUNTER2", None, ""),
    (about_them, "MY NAME IS LUCA, MY PHONE IS 3331234567", None, "MY NAME IS LUCA,"),
    (about_them, "I WORK ON STEM CELLS", None, "I WORK ON STEM CELLS"),
    (drop_sensitive, "Maria likes optics. Her password is HUNTER2.", None, "Maria likes optics."),
    (looks_like_a_name, "Alessandro's Student", None, False),
    (looks_like_a_name, "Anna Maria", None, True),
    (introduction, "HELLO, I'M ALEJANDRO HOW ARE YOU?", None, "I'M ALEJANDRO"),
    (introduction, "HELLO, I\u2019M ALEJANDRO", None, "I'M ALEJANDRO"),
    (introduction, "MY NAME IS ANNA MARIA AND I LIKE OPTICS", None, "MY NAME IS ANNA MARIA"),
    (introduction, "WHAT IS MY NAME?", None, ""),
    (is_self_introduction, "HELLO, I’M ALEJANDRO", "Alejandro", True),
    # the model's own reasoning written into the notes
    (drop_bookkeeping, "Davide passed his driving test. (Previously known: Davide is preparing "
     "for his driving test - this is no longer true.)", None, "Davide passed his driving test."),
    (drop_bookkeeping, "Davide passed his driving test. Previously known fact 'Davide is "
     "preparing for his driving test' is no longer true.", None, "Davide passed his driving test."),
    (drop_bookkeeping, "Elena signed the lease for her new flat. Previously known information "
     "about her looking for an apartment in Turin is no longer accurate.", None,
     "Elena signed the lease for her new flat."),
    (drop_bookkeeping, "Anna is writing a paper on depth sensing (this remains true as the new "
     "LIDAR project aligns with depth sensing research). Anna started a new project on LIDAR.",
     None, "Anna is writing a paper on depth sensing. Anna started a new project on LIDAR."),
    # facts, even about something that ended, stay
    (drop_bookkeeping, "Giulia is no longer writing her thesis, as she has handed it in.", None,
     "Giulia is no longer writing her thesis, as she has handed it in."),
    (drop_bookkeeping, "Paolo plays chess. Paolo lives in Turin now.", None,
     "Paolo plays chess. Paolo lives in Turin now."),
    (drop_bookkeeping, "Luca works on depth sensing (stereo cameras).", None,
     "Luca works on depth sensing (stereo cameras)."),
    (drop_bookkeeping, "Studies chemistry. No other information known.", None, "Studies chemistry."),
    (drop_bookkeeping, "Alessandro is building a robot arm. The encounter did not provide new "
     "information to contradict this fact.", None, "Alessandro is building a robot arm."),
    (drop_reported, "Luca studies chemistry. Initially indicated they study physics, but later "
     "clarified that their field is chemistry.", None, "Luca studies chemistry."),
    (drop_reported, "They study chemistry. They have corrected a previous statement about "
     "studying physics.", None, "They study chemistry."),
    (drop_reported, "They study chemistry, having corrected an initial statement about studying "
     "physics.", None, "They study chemistry."),
    (drop_reported, "Luca works in the robotics group. Luca previously worked in the vision group.",
     None, "Luca works in the robotics group. Luca previously worked in the vision group."),
]


def main():
    failed = 0
    for (text, name), want in CASES:
        got = unshout(text, name)
        ok = got == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {text!r:48} -> {got!r}" + ("" if ok else f"  (expected {want!r})"))
    for text, name, want in NAME_CASES:
        got = drop_name_sentences(text, name)
        ok = got == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {text!r:48} -> {got!r}")
    for fn, text, arg, want in RULE_CASES:
        got = fn(text, arg) if arg is not None else fn(text)
        ok = got == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {fn.__name__:20} {text!r:40} -> {got!r}")
    # A careful write-up with no notes must not wipe what was known.
    import two_stage as T
    real = (T.chat_stream, T.chat)
    for stream, fast, old, want in (
            ('{"summary":"x"}', '{"summary":"x","notes":"Maria is writing her thesis."}', "",
             "Maria is writing her thesis."),
            ('{"summary":"x"}', '{"summary":"x"}', "Alessandro is building a robot arm.",
             "Alessandro is building a robot arm."),
            ('{"summary":"x","notes":"Luca studies chemistry."}', '{}', "", "Luca studies chemistry.")):
        T.chat_stream = lambda *a, _r=stream, **k: _r
        T.chat = lambda *a, _r=fast, **k: _r
        got = T.summarise_encounter("Maria", old, [("I AM WRITING MY THESIS", "")], careful=True)[1]
        ok = got == want
        failed += not ok
        RULE_CASES.append(None)
        print(f"  {'PASS' if ok else 'FAIL'}  no notes returned {old!r:34} -> {got!r}")
    T.chat_stream, T.chat = real
    RULE_CASES[:] = [c for c in RULE_CASES if c is not None]
    total = len(CASES) + len(NAME_CASES) + len(RULE_CASES) + 3
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
