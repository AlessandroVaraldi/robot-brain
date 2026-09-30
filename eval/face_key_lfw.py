#!/usr/bin/env python3
"""pi/face_key end to end on LFW (press photos, not the robot's camera).

Half of the people with three frontal photos or more are enrolled from their
first two; the other half never are. A face is recognised from two frames,
each of which must open a lock of the same person. Then:

  enrol     does anyone, enrolling, turn out to be someone already enrolled?
  genuine   every pair of further photos (up to five) of an enrolled person:
            recognised as themselves, as someone else, or not at all
  visit     people with two further photos or more: recognised from some pair
            of consecutive ones, as a visit with several frames would be
  strangers every pair of photos (up to five) of the people never enrolled
  refresh   people with seven photos or more: pairs from photos 5-7, then
            again after a new lock was made from photos 3-4 a day later

Declared before running: at least 90% of genuine pairs and 95% of visits
recognised; nobody taken for someone else, at enrolment or after; no stranger
recognised; and two frames checked against everyone enrolled in at most 50 ms
on the machine running this.

    python3 eval/face_key_lfw.py
"""

import itertools
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pi"))
sys.path.insert(0, str(HERE))
from face_bits import embeddings  # noqa: E402
from face_key.chip import SoftChip  # noqa: E402
from face_key.store import FaceKeys, admin_keypair  # noqa: E402


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def new_store(tmp, clock):
    return FaceKeys(SoftChip(key=os.urandom(32), burst=10**9), admin_keypair()[1],
                    Path(tmp) / "faces.sqlite", clock=clock)


def main():
    names, V = embeddings()
    by = {}
    for i, n in enumerate(names):
        by.setdefault(n, []).append(i)
    people = sorted(n for n, ix in by.items() if len(ix) >= 3)
    random.Random(0).shuffle(people)
    enrolled, strangers = people[::2], people[1::2]
    print(f"enrolled {len(enrolled)}, strangers {len(strangers)}")

    with tempfile.TemporaryDirectory() as tmp:
        keys = new_store(tmp, Clock())
        keys.REFRESH_AFTER_S = float("inf")          # refresh measured on its own below
        t = time.perf_counter()
        ids = {n: keys.enrol([V[i] for i in by[n][:2]], n).id for n in enrolled}
        print(f"enrol: {(time.perf_counter() - t) / len(enrolled) * 1e3:.1f} ms per person; "
              f"{len(enrolled) - len(set(ids.values()))} taken for someone already enrolled")

        right = wrong = pairs = visits = could_visit = 0
        times = []
        for n in enrolled:
            probes = by[n][2:7]
            for a, b in itertools.combinations(probes, 2):
                t = time.perf_counter()
                got = keys.recognise([V[a], V[b]])
                times.append(time.perf_counter() - t)
                pairs += 1
                right += bool(got and got.id == ids[n])
                wrong += bool(got and got.id != ids[n])
            if len(probes) >= 2:
                could_visit += 1
                visits += any((got := keys.recognise([V[a], V[b]])) and got.id == ids[n]
                              for a, b in zip(probes, probes[1:]))
        accepted = tried = 0
        for n in strangers:
            for a, b in itertools.combinations(by[n][:5], 2):
                tried += 1
                accepted += keys.recognise([V[a], V[b]]) is not None

    print(f"genuine pairs: {right}/{pairs} = {right / pairs:.1%} recognised, "
          f"{wrong} taken for someone else")
    print(f"visits: {visits}/{could_visit} = {visits / could_visit:.1%}")
    print(f"strangers: {accepted} of {tried} pairs recognised as someone")
    print(f"time for two frames against {len(enrolled)} people: median "
          f"{np.median(times) * 1e3:.1f} ms, max {max(times) * 1e3:.1f} ms")

    many = [n for n in people if len(by[n]) >= 7]
    with tempfile.TemporaryDirectory() as tmp:
        clock = Clock()
        keys = new_store(tmp, clock)
        ids = {n: keys.enrol([V[i] for i in by[n][:2]], n).id for n in many}

        def late():
            return [bool((got := keys.recognise([V[a], V[b]])) and got.id == ids[n])
                    for n in many for a, b in itertools.combinations(by[n][4:7], 2)]

        before = late()                     # same day: no new locks yet
        clock.t += 86400
        refreshed = sum(bool(keys.recognise([V[i] for i in by[n][2:4]])) for n in many)
        clock.t += 60                       # the late pairs must not refresh again
        keys.REFRESH_AFTER_S = float("inf")
        after = late()
    print(f"refresh ({len(many)} people, {refreshed} got a new lock): pairs from photos 5-7 "
          f"recognised {np.mean(before):.1%} before, {np.mean(after):.1%} after")


if __name__ == "__main__":
    main()
