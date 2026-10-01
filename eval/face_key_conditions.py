#!/usr/bin/env python3
"""Face keys under other light and other cameras: LFW photos, changed as a day
or a camera would change them, through the whole path (YuNet, SFace and its
frontal filter, the locks).

Half of the people with four frontal photos or more are enrolled from their
first two photos, untouched; the other half never are. Their further photos
(3 to 7) are changed the same way, and handed over two at a time, as the
robot hands over two frames:

  found       photos 3 and 4 both still give a frontal face large enough
  pair        enrolled people recognised from photos 3 and 4
  visit       ... from some consecutive pair of photos 3 to 7, as a visit
              with several frames would be
  wrong       enrolled people taken for someone else (any pair)
  strangers   never-enrolled people recognised as someone (any pair)

Declared before running: one stop darker (a cloudy day) loses at most 5
points of recognition against the untouched photos, and no change makes a
stranger or the wrong person recognised.

    python3 eval/face_key_conditions.py
    python3 eval/face_key_conditions.py --only light

CPU-heavy: run it on a machine of your own, not a shared server.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "john"))
sys.path.insert(0, str(HERE.parent / "pi"))
from paths import LFW  # noqa: E402

# Leave room for everything else on the machine (on a shared server, run it elsewhere).
WORKERS = max(1, (os.cpu_count() or 2) - 2)

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


LIGHT = ("1 stop darker", "2 stops darker", "3 stops darker", "overexposed", "low contrast",
         "light from one side", "dim and noisy", "noise")


def embed(job):
    global _fc
    path, name, seed = job
    if _fc is None:
        import cv2
        cv2.setNumThreads(1)          # one process per core already
        from face_compare import FaceComparer
        _fc = FaceComparer()
    vec = _fc.analyse(changed(path, name, seed))["vec"]
    return None if vec is None else np.asarray(vec, np.float32).ravel()


def frontal_people():
    """name -> frontal photos, untouched; cached, keyed by
    the face code that decides what is frontal."""
    code = (HERE.parent / "john" / "face_compare.py").read_bytes()
    cache = HERE / "suite_out" / f"lfw_frontal_{hashlib.sha256(code).hexdigest()[:10]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    files = sorted((LFW / "lfw").glob("*/*.jpg"))
    with ProcessPoolExecutor(WORKERS) as ex:
        vecs = list(ex.map(embed, [(f, "untouched", 0) for f in files], chunksize=64))
    by = {}
    for f, v in zip(files, vecs):
        if v is not None:
            by.setdefault(f.parent.name, []).append(str(f))
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(by))
    print(f"LFW: {len(files)} photos, {sum(map(len, by.values()))} frontal and large enough")
    return by


def run(by, people, enrolled, strangers, conditions):
    from face_key.chip import SoftChip
    from face_key.store import FaceKeys, admin_keypair
    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(WORKERS) as ex:
        keys = FaceKeys(SoftChip(key=os.urandom(32), burst=10**9), admin_keypair()[1],
                        Path(tmp) / "faces.sqlite")
        keys.REFRESH_AFTER_S = float("inf")
        enrol = list(ex.map(embed, [(by[n][i], "untouched", 0)
                                    for n in enrolled for i in (0, 1)], chunksize=16))
        ids = {}
        for k, n in enumerate(enrolled):
            pair = [v for v in enrol[2 * k:2 * k + 2] if v is not None]
            if len(pair) == 2:
                ids[n] = keys.enrol(pair, n).id
        print(f"\nenrolled {len(ids)} of {len(enrolled)}, strangers {len(strangers)}")
        print(f"{'condition':22} {'found':>6} {'pair':>7} {'visit':>7} {'wrong':>6} {'strangers':>10}")
        pairs = {}
        for name in conditions:
            jobs = [(by[n][i], name, zlib.crc32(f"{n}:{i}".encode()))
                    for n in people for i in range(2, min(7, len(by[n])))]
            vecs = iter(ex.map(embed, jobs, chunksize=16))
            probes = {n: [next(vecs) for _ in range(2, min(7, len(by[n])))] for n in people}

            def opens(a, b):
                return keys.recognise([a, b]) if a is not None and b is not None else None
            found = sum(p[0] is not None and p[1] is not None for p in probes.values())
            pair = visit = wrong = accepted = 0
            for n in ids:
                p = probes[n]
                got = [opens(a, b) for a, b in zip(p, p[1:])]
                pair += bool(got[0] and got[0].id == ids[n])
                visit += any(g and g.id == ids[n] for g in got)
                wrong += sum(bool(g and g.id != ids[n]) for g in got)
            for n in strangers:
                p = probes[n]
                accepted += sum(bool(opens(a, b)) for a, b in zip(p, p[1:]))
            pairs[name] = pair / len(ids)
            print(f"{name:22} {found / len(people):>6.0%} {pair / len(ids):>7.1%} "
                  f"{visit / len(ids):>7.1%} {wrong:>6} {accepted:>10}", flush=True)
        light_ones = [pairs[c] for c in LIGHT if c in pairs]
        if light_ones:
            print(f"mean pair recognition over the {len(light_ones)} light conditions: "
                  f"{np.mean(light_ones):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["light"], help="untouched and the light conditions only")
    a = ap.parse_args()
    conditions = ["untouched", *LIGHT] if a.only == "light" else list(CONDITIONS)
    by = frontal_people()
    people = sorted(n for n, ph in by.items() if len(ph) >= 4)
    random.Random(0).shuffle(people)
    enrolled, strangers = people[::2], people[1::2]
    run(by, people, enrolled, strangers, conditions)


if __name__ == "__main__":
    main()
