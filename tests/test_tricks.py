#!/usr/bin/env python3
"""tricks.py: what fires on which writing, and whether the answer is exact.
No models."""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
from tricks import Fact, said, sequence, work_out  # noqa: E402

TODAY = dt.date(2026, 9, 30)             # a Wednesday

FIRES = [  # writing, what the fact must contain
    ("I WAS BORN ON 12 MARCH 1990", ("Monday", "then: 36")),
    ("WHAT DAY WAS 25/12/1999?", ("Saturday",)),
    ("WHAT DAY OF THE WEEK WAS JULY 20, 1969?", ("Sunday",)),
    ("1990-03-12", ("Monday",)),
    ("WHAT DAY WAS 03/04/2020?", ("3 April 2020", "Friday")),   # dd/mm/yyyy first
    ("12/31/1999", ("Friday",)),                  # month first only when day first is impossible
    ("MY BIRTHDAY IS JULY 4TH", ("Sunday", "277 days")),
    ("MY BIRTHDAY IS 30 SEPTEMBER", ("today", "Wednesday")),
    ("MY BIRTHDAY IS 29 FEBRUARY", ("2028", "Tuesday")),
    ("WHAT DAY WILL 1 JANUARY 2030 BE?", ("will be", "Tuesday")),
    ("WHAT IS 1234 X 5678?", ("7006652",)),
    ("1234 × 5678", ("7006652",)),
    ("WHAT IS 2+2?", ("= 4",)),
    ("WHAT IS 7 TIMES 8?", ("56",)),
    ("WHAT IS 10/4?", ("2.5",)),
    ("WHAT IS 144 / 12?", ("= 12",)),
    ("2 TO THE POWER OF 64", ("18446744073709551616",)),
    ("WHAT IS 3 SQUARED?", ("= 9",)),
    ("IS 97 PRIME?", ("97 is a prime",)),
    ("IS 91 PRIME?", ("not prime", "7 × 13")),
    ("PRIME FACTORS OF 360", ("2 × 2 × 2 × 3 × 3 × 5",)),
    ("WHAT IS THE SQUARE ROOT OF 144?", ("12, exactly",)),
    ("SQUARE ROOT OF 2", ("1.41421",)),
    ("IS 1000000000039 PRIME?", ("is a prime",)),                  # beyond trial division
    ("IS 999999999999999989 PRIME?", ("is a prime",)),             # the largest 18-digit prime
    ("IS 1000000000000000003 PRIME?", ("too big",)),               # beyond the limit
    ("IS 0.5 PRIME?", ("not a whole number",)), ("IS 1/2 PRIME?", ("not a whole number",)),
    ("IS PI PRIME?", ("π is not a whole number",)), ("IS -7 PRIME?", ("positive",)),
    ("IS 0 PRIME?", ("two divisors",)), ("IS 7.0 PRIME?", ("7 is a prime",)),
    ("IS SEVEN PRIME?", ("7 is a prime",)), ("IS 1,000,003 PRIME?", ("1000003 is a prime",)),
    ("PRIME FACTORS OF 2.5", ("not a whole number",)), ("PRIME FACTORS OF 1", ("no prime factors",)),
    ("FACTOR 10000000000001", ("too big",)),
    ("SQUARE ROOT OF -4", ("±2i",)), ("SQUARE ROOT OF 0.25", ("0.5, exactly",)),
]

SILENT = [  # nothing to work out, or not asked for
    "WHAT IS MY NAME?", "ROOM 42 AT 3PM", "THINK OF A NUMBER 1-100", "MY PHONE IS 3331234567",
    "I AM 25 YEARS OLD", "WHAT DAY IS IT TODAY?", "WHAT IS 1/0?", "WHAT IS 9**9**9?",
    "42", "HELLO", "", "I WAS BORN ON 31 FEBRUARY 1990", "MARCH 1066",
]

# One conversation, board by board: (writing, the list held after it, what the
# fact must contain, or None for no fact).
CONVERSATION = [
    ("WHAT IS 2+2?", [], None),                                  # no list yet: not ours
    ("REMEMBER: APPLE, TRAIN, 42, BLUE", ["Apple", "Train", "42", "Blue"], "4 things"),
    ("HOW MANY PEOPLE DID YOU MEET TODAY?", None, None),         # not about the list
    ("ALSO REMEMBER 7", ["Apple", "Train", "42", "Blue", "7"], "5 in all"),
    ("WHAT DID I ASK YOU TO REMEMBER?", None, "Apple, Train, 42, Blue, 7"),
    ("NOW SAY IT BACKWARDS", None, "7, Blue, 42, Train, Apple"),
    ("WHAT WAS THE THIRD ONE?", None, "42"),
    ("AND THE LAST ONE?", None, ": 7"),
    ("WHAT WAS THE 9TH ITEM?", None, "no number 9"),
    ("HOW MANY THINGS DID I GIVE YOU?", None, "5 things"),
    ("SORT THEM", None, "7, 42, Apple, Blue, Train"),
    ("WHAT IS THE FIRST MONTH OF THE YEAR?", None, None),        # an ordinal, not about the list
    ("REMEMBER 9 1 4 7", ["9", "1", "4", "7"], "4 things"),      # a new list replaces the old
    ("REPEAT IT", None, "9, 1, 4, 7"),
    ("REMEMBER THAT ALESSANDRO IS A THIEF", ["Alessandro is a thief"], "1 thing"),   # a sentence is one item
    ("REMEMBER: " + ", ".join(f"W{i}" for i in range(40)), [f"W{i}" for i in range(30)], "Holding 30"),
]


# Anagrams on a small made-up vocabulary, so the test needs neither wordfreq nor
# a word list: (writing, their name, what the fact must contain).
FREQ = {"silent": 4.5, "listen": 5.0, "enlist": 3.5, "love": 6.0, "air": 5.5, "role": 5.0,
        "via": 5.0, "alive": 5.0, "or": 7.0, "ai": 4.0, "arm": 4.5, "am": 6.0,
        "reads": 5.0, "loans": 4.8, "roads": 4.9, "lanes": 4.3}
ANAGRAMS = [
    ("ANAGRAM OF LISTEN", "", "Silent"),
    ("CAN YOU MAKE AN ANAGRAM OF THE WORD LISTEN?", "", "Silent"),
    ("ANAGRAM MY NAME", "Alessandro", "Reads loans"),           # the pair of commonest words
    ("ANAGRAM MY NAME", "Maria", "Air am"),               # not "AI ARM": AI is not a word to say
    ("ANAGRAM MY NAME", "", "not told you their name"),
    ("ANAGRAM OF XYZQ", "", "No English word"),
    ("ANAGRAM OF SUPERCALIFRAGILISTIC", "", "too long"),
    ("WHAT IS AN ANAGRAM?", "", None),                    # a question about anagrams
]


def anagram_cases():
    import tricks
    tricks._words = {}
    for w in FREQ:
        tricks._words.setdefault("".join(sorted(w)), []).append(w)
    for ws in tricks._words.values():
        ws.sort(key=lambda w: -FREQ[w])
    tricks._zipf = FREQ.get
    failed = 0
    for writing, name, part in ANAGRAMS:
        f = tricks.anagram(writing, name)
        ok = (f is None) if part is None else (f is not None and part in f.text)
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {writing + ' / ' + name:40} -> {f.text if f else 'nothing'}")
    return failed


def main():
    failed = 0
    held = []
    for writing, want_held, part in CONVERSATION:
        before = held
        held, f = sequence(writing, held)
        ok = (held == (want_held if want_held is not None else before)
              and ((f is None) if part is None else (f is not None and part in f.text)))
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {writing:40} -> {f.text[:70] if f else 'nothing'}")
    held2, f = sequence("WHAT DID I ASK YOU TO REMEMBER?", [])
    ok = held2 == [] and f is not None and "not given you anything" in f.text
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  nothing held: says so, invents nothing")
    ok = sequence("REPEAT IT", [])[1] is None
    failed += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  REPEAT IT with nothing held: not the list's")
    order = Fact("backwards", "7 | blue | 42")
    for reply, want in (("7, blue, 42.", True), ("Seven... no: 7, Blue and 42!", True),
                        ("42, blue, 7.", False)):
        ok = said(order, reply) == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  said(in order, {reply!r}) == {want}")
    for writing, parts in FIRES:
        f = work_out(writing, TODAY)
        ok = f is not None and all(p in f.text for p in parts) and said(f, f.text)
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {writing:40} -> {f.text if f else None}")
    for writing in SILENT:
        f = work_out(writing, TODAY)
        ok = f is None
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {writing!r:40} -> {f.text if f else 'nothing'}")
    fact = Fact("1234 × 5678 = 7006652.", "7006652")
    for reply, want in (("That's 7,006,652.", True), ("That's 7 006 652.", True),
                        ("Roughly seven million.", False)):
        ok = said(fact, reply) == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  said({reply!r}) == {want}")
    prime = work_out("IS 1000000000039 PRIME?", TODAY)
    for reply, want in (("Yes, 1000000000039 is prime.", True),
                        ("No, 1000000000039 is not prime; it factors into itself.", False),
                        ("It isn't a prime.", False)):
        ok = said(prime, reply) == want
        failed += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  prime: said({reply!r}) == {want}")
    failed += anagram_cases()
    total = len(FIRES) + len(SILENT) + 3 + 3 + len(CONVERSATION) + 3 + len(ANAGRAMS) + 2
    print(f"\n{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
