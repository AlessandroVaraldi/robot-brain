#!/usr/bin/env python3
"""FaceComparer on LFW pairs (press photos, not our camera) and on our frames.

Checks the plumbing and the frontal filter; the threshold for the robot comes
from validate_faces.py.  Declared: on frontal LFW pairs, no different pair
taken for the same, and at least 90% of same pairs recognised.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "john"))
from face_compare import FaceComparer  # noqa: E402

from paths import FRAMES, LFW  # noqa: E402


def pairs():
    lines = (LFW / "pairsDevTest.txt").read_text().split("\n")
    n = int(lines[0])
    for l in lines[1:1 + n]:
        a, i, j = l.split()
        yield (a, i), (a, j), True
    for l in lines[1 + n:1 + 2 * n]:
        a, i, b, j = l.split()
        yield (a, i), (b, j), False


def main():
    # LFW faces are ~100px in a 250px photo: min_width lowered for them only.
    fc = FaceComparer(min_width=40)
    cache = {}

    def vec(name, idx):
        if (name, idx) not in cache:
            p = LFW / "lfw" / name / f"{name}_{int(idx):04d}.jpg"
            cache[(name, idx)] = fc.embed(p.read_bytes()) if p.exists() else None
        return cache[(name, idx)]

    same_ok = same_n = diff_bad = diff_n = skipped = 0
    for a, b, same in pairs():
        va, vb = vec(*a), vec(*b)
        if va is None or vb is None:
            skipped += 1
            continue
        s = fc.same(va, vb)
        if same:
            same_n += 1
            same_ok += s
        else:
            diff_n += 1
            diff_bad += s
    print(f"LFW, frontal pairs only ({skipped} skipped: face turned or small)")
    print(f"   same person recognised: {same_ok}/{same_n} ({same_ok / same_n:.0%})")
    print(f"   different people taken for the same: {diff_bad}/{diff_n}")

    fc = FaceComparer()
    print("\nour frames (frontal filter):")
    for name in ("pose1", "pose2", "pose3", "real_board"):
        a = fc.analyse((FRAMES / f"{name}.jpg").read_bytes())
        w = f"{a['width']:.0f}px yaw {a['yaw']:.2f}" if a.get("faces") else ""
        print(f"   {name:11} {w:20} {'usable' if a['vec'] is not None else a['why']}")

    ok = diff_bad == 0 and same_ok / same_n >= 0.9
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
