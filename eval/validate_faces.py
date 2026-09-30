#!/usr/bin/env python3
"""Validate the face comparison on the robot's own camera.

Four short sessions, one person at a time in front of the robot, looking at it:

    A1  person A        B1  person B        A2  person A again   B2  person B again

(changing distance or light a little between A1 and A2 makes it a better test).
Each session takes --per-session snapshots, one a second.  Then:

* same person (A with A, B with B, across sessions): how often they come out
  "same" - a miss splits one conversation in two, which costs little;
* different people (A with B): how often they come out "same" - that would
  hand one person's memory to another, and must be zero;
* the threshold these scores suggest.

Read-only on the robot: it GETs /status and /snapshot.jpg and nothing else - no
arm, no tracking, no speech.  No image and no vector is saved: everything stays
in RAM and is gone when the script ends.

    python3 validate_faces.py                    # the real session
    python3 validate_faces.py --from-files A:frames/pose1.jpg B:...   # offline
"""

import argparse
import itertools
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
from face_compare import FaceComparer  # noqa: E402

BRIDGE = "http://192.168.50.2:8088"


def get(path, timeout=10):
    with urllib.request.urlopen(f"{BRIDGE}{path}", timeout=timeout) as r:
        return r.read()


def live_session(fc, label, n, interval):
    input(f"\n[{label}] person {label[0]} in front of the robot, looking at the camera. Enter to start... ")
    got = []
    for i in range(n):
        t = time.time()
        a = fc.analyse(get("/snapshot.jpg?w=640&q=90"))
        show(label, i, a)
        if a["vec"] is not None:
            got.append(a["vec"])
        time.sleep(max(0.0, interval - (time.time() - t)))
    return got


def show(label, i, a):
    if a.get("faces"):
        extra = f"width {a['width']:.0f}px yaw {a['yaw']:.2f}"
    else:
        extra = ""
    state = "usable" if a["vec"] is not None else f"skipped: {a['why']}"
    print(f"   {label} #{i + 1:<2} faces {a.get('faces', 0)}  {extra:34} {state}")


def report(fc, sessions):
    by_person = {}
    for label, vecs in sessions.items():
        by_person.setdefault(label[0], []).extend(vecs)
    for p, v in by_person.items():
        print(f"person {p}: {len(v)} usable frames")
    if len(by_person) < 2 or any(len(v) < 2 for v in by_person.values()):
        raise SystemExit("need at least 2 usable frames for each of two people")

    same = [fc.score(a, b) for v in by_person.values() for a, b in itertools.combinations(v, 2)]
    diff = [fc.score(a, b) for (pa, va), (pb, vb) in itertools.combinations(by_person.items(), 2)
            for a in va for b in vb]
    same, diff = np.array(same), np.array(diff)
    thr = fc.threshold
    print(f"\nsame person:       {len(same)} pairs  min {same.min():+.3f}  median {np.median(same):+.3f}")
    print(f"different people:  {len(diff)} pairs  max {diff.max():+.3f}  median {np.median(diff):+.3f}")
    print(f"\nat the current threshold {thr:.3f}:")
    print(f"   same person taken for different: {(same < thr).mean():.0%}  (splits a conversation)")
    print(f"   different people taken for the same: {(diff >= thr).mean():.0%}  (must be 0)")
    if diff.max() < same.min():
        mid = (diff.max() + same.min()) / 2
        print(f"\nclean separation: margin {same.min() - diff.max():.3f}. "
              f"Suggested threshold {mid:.3f} (halfway between the two groups).")
    else:
        safe = diff.max() + 0.02
        print(f"\ngroups overlap. Safe threshold {safe:.3f} (above every pair of different "
              f"people): same person recognised {(same >= safe).mean():.0%}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-session", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--from-files", nargs="*", help="LABEL:path ... for an offline run")
    a = ap.parse_args()
    fc = FaceComparer()
    sessions = {}
    if a.from_files:
        for i, item in enumerate(a.from_files):
            label, path = item.split(":", 1)
            an = fc.analyse(Path(path).read_bytes())
            show(label, i, an)
            if an["vec"] is not None:
                sessions.setdefault(label, []).append(an["vec"])
    else:
        st = json.loads(get("/status"))
        print(f"bridge reachable; tracking: {st.get('tracking', {}).get('mode', '?')} "
              "(this script does not touch it)")
        for label in ("A1", "B1", "A2", "B2"):
            sessions[label] = live_session(fc, label, a.per_session, a.interval)
    report(fc, sessions)


if __name__ == "__main__":
    main()
