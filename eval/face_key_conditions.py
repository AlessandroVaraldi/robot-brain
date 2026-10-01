#!/usr/bin/env python3
"""Face keys under other light and other cameras: LFW photos, changed as a day
or a camera would change them, through the whole path (YuNet, SFace and its
frontal filter, the locks).

Half of the people with four frontal photos or more are enrolled from their
first two photos, untouched; the other half never are. Then photos 3 and 4 of
everyone are changed the same way and handed over as a pair, as the robot
hands over two frames:

  found       both photos still give a frontal face large enough to use
  recognised  enrolled people recognised as themselves (of all their pairs)
  wrong       enrolled people taken for someone else
  strangers   pairs of the never-enrolled recognised as someone

Declared before running: one stop darker (a cloudy day) loses at most 5
points of recognition against the untouched photos, and no change makes a
stranger or the wrong person recognised.

    python3 eval/face_key_conditions.py
"""

import io
import os
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "john"))
sys.path.insert(0, str(HERE.parent / "pi"))
from paths import LFW  # noqa: E402

_fc = None


def side_light(img, low=0.3):
    """Light from one side: the far side of the face down to `low`."""
    w = img.shape[1]
    return img * np.linspace(1.0, low, w)[None, :, None]


CONDITIONS = {
    "untouched": lambda im, r: im,
    "1 stop darker": lambda im, r: im * 0.5,
    "2 stops darker": lambda im, r: im * 0.25,
    "3 stops darker": lambda im, r: im * 0.125,
    "overexposed": lambda im, r: im * 1.6,
    "low contrast": lambda im, r: (im - im.mean()) * 0.5 + im.mean(),
    "light from one side": lambda im, r: side_light(im),
    "warm light": lambda im, r: im * np.array([0.75, 0.95, 1.15]),     # BGR
    "cold light": lambda im, r: im * np.array([1.15, 1.0, 0.8]),
    "dim and noisy": lambda im, r: im * 0.35 + r.normal(0, 8, im.shape),
    "noise": lambda im, r: im + r.normal(0, 15, im.shape),
    "slight blur": "blur 1.5",
    "blur": "blur 3",
    "jpeg 20": "jpeg 20",
    "farther (0.7x)": "scale 0.7",
    "head tilted 15 deg": "rotate 15",
}


def changed(path, name, seed):
    import cv2
    img = cv2.imread(str(path)).astype(np.float32)
    how = CONDITIONS[name]
    quality = 92
    if callable(how):
        img = how(img, np.random.default_rng(seed))
    else:
        kind, x = how.split()
        x = float(x)
        if kind == "blur":
            img = cv2.GaussianBlur(img, (0, 0), x)
        elif kind == "jpeg":
            quality = int(x)
        elif kind == "scale":
            img = cv2.resize(img, None, fx=x, fy=x, interpolation=cv2.INTER_AREA)
        elif kind == "rotate":
            h, w = img.shape[:2]
            img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), x, 1.0), (w, h),
                                 borderMode=cv2.BORDER_REFLECT)
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def embed(job):
    global _fc
    if _fc is None:
        import cv2
        cv2.setNumThreads(1)          # one process per core already
        from face_compare import FaceComparer
        _fc = FaceComparer()
    path, name, seed = job
    vec = _fc.analyse(changed(path, name, seed))["vec"]
    return None if vec is None else np.asarray(vec, np.float32).ravel()


def main():
    from face_key.chip import SoftChip
    from face_key.store import FaceKeys, admin_keypair

    files = sorted((LFW / "lfw").glob("*/*.jpg"))
    with ProcessPoolExecutor(16) as ex:
        base = list(ex.map(embed, [(f, "untouched", 0) for f in files], chunksize=64))
    by = {}
    for f, v in zip(files, base):
        if v is not None:
            by.setdefault(f.parent.name, []).append((f, v))
    print(f"LFW: {len(files)} photos, {sum(map(len, by.values()))} frontal and large enough "
          f"({sum(map(len, by.values())) / len(files):.0%}); the others never reach the locks")
    people = sorted(n for n, ph in by.items() if len(ph) >= 4)
    random.Random(0).shuffle(people)
    enrolled, strangers = people[::2], people[1::2]

    with tempfile.TemporaryDirectory() as tmp:
        keys = FaceKeys(SoftChip(key=os.urandom(32), burst=10**9), admin_keypair()[1],
                        Path(tmp) / "faces.sqlite")
        keys.REFRESH_AFTER_S = float("inf")
        ids = {n: keys.enrol([v for _, v in by[n][:2]], n).id for n in enrolled}
        print(f"enrolled {len(enrolled)}, strangers {len(strangers)}; probes: photos 3 and 4\n")
        print(f"{'condition':22} {'found':>7} {'recognised':>11} {'wrong':>6} {'strangers':>10}")
        for name in CONDITIONS:
            jobs = [(by[n][i][0], name, hash((n, i)) % 2**31) for n in people for i in (2, 3)]
            with ProcessPoolExecutor(16) as ex:
                vecs = list(ex.map(embed, jobs, chunksize=16))
            pair = {n: (vecs[2 * k], vecs[2 * k + 1]) for k, n in enumerate(people)}
            found = sum(a is not None and b is not None for a, b in pair.values())
            right = wrong = accepted = 0
            for n in enrolled:
                a, b = pair[n]
                got = keys.recognise([a, b]) if a is not None and b is not None else None
                right += bool(got and got.id == ids[n])
                wrong += bool(got and got.id != ids[n])
            for n in strangers:
                a, b = pair[n]
                accepted += bool(a is not None and b is not None and keys.recognise([a, b]))
            print(f"{name:22} {found / len(people):>7.0%} {right / len(enrolled):>11.1%} "
                  f"{wrong:>6} {accepted:>10}", flush=True)


if __name__ == "__main__":
    main()
