#!/usr/bin/env python3
"""The frontal filter of face_compare.py on made-up landmarks: a head tilted
in the picture but looking at the camera is frontal, a head turned is not.
No models."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
from face_compare import yaw_of  # noqa: E402

RESULTS = []


def check(ok, label):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


def row(turn=0.0, tilt_deg=0.0):
    """YuNet's row: box, then right eye, left eye, nose (x, y).  `turn` moves
    the nose sideways in eye-distances; the face is then tilted in the picture."""
    pts = np.array([[-20.0, 0.0], [20.0, 0.0], [40.0 * turn, 25.0]])
    a = np.radians(tilt_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = pts @ rot.T + [100.0, 100.0]
    return [60, 60, 80, 80, *pts.ravel()]


def main():
    check(yaw_of(row()) == 0.0, "frontal, upright: 0")
    for tilt in (10, 15, 25, -20):
        check(yaw_of(row(tilt_deg=tilt)) < 1e-9, f"frontal, tilted {tilt} degrees: still 0")
    check(abs(yaw_of(row(turn=0.3)) - 0.3) < 1e-9, "turned 0.3: 0.3, as before")
    check(abs(yaw_of(row(turn=0.3, tilt_deg=20)) - 0.3) < 1e-9, "turned and tilted: the turn")
    eyes_on_top = [60, 60, 80, 80, 100, 100, 100, 100, 110, 120]
    check(yaw_of(eyes_on_top) == 9.9, "eyes in one place: not frontal")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
