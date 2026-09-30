#!/usr/bin/env python3
"""Can a face be turned into bits stable enough to derive a key from?

On LFW (press photos, not the robot's camera), with the robot's face model
(SFace, frontal faces only, as face_compare.py): the fraction of bits that
change between two photos of the same person (genuine) and of two different
people (impostor), for several ways of turning a 128-number face vector into
bits.  A key can be rebuilt from a face if an error-correcting code absorbs
the genuine changes while impostors stay near 50%.

  cosine    the continuous comparison used today, for reference
  sign128   one bit per dimension
  proj512   512 random projections
  reliable  1024 projections, keeping the 256 furthest from zero at enrolment
            (which ones is stored as helper data)
  multi     as reliable, enrolled from the mean of two photos

Declared before running: feasible if a method keeps impostors below 1 in
10,000 while recognising at least 90% of genuine pairs, at a bit-error
threshold a real code can correct (roughly under 10-15%).

    python3 eval/face_bits.py
"""

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "john"))
from paths import LFW  # noqa: E402

_fc = None


def embed(path):
    global _fc
    if _fc is None:
        from face_compare import FaceComparer
        _fc = FaceComparer()
    vec = _fc.analyse(Path(path).read_bytes())["vec"]
    return None if vec is None else np.asarray(vec, dtype=np.float32).ravel()


def embeddings():
    cache = HERE / "suite_out" / "lfw_sface.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["names"]), z["vecs"]
    paths = sorted((LFW / "lfw").glob("*/*.jpg"))
    with ProcessPoolExecutor(16) as ex:
        vecs = list(ex.map(embed, map(str, paths), chunksize=64))
    keep = [(p.parent.name, v) for p, v in zip(paths, vecs) if v is not None]
    print(f"faces: {len(paths)}, frontal and large enough: {len(keep)}")
    names, arr = [k for k, _ in keep], np.stack([v for _, v in keep])
    np.savez(cache, names=np.array(names, dtype=object), vecs=arr)
    return names, arr


def pairs(names, rng, per_person=5, impostors=40):
    by = {}
    for i, n in enumerate(names):
        by.setdefault(n, []).append(i)
    people = [n for n, ix in by.items() if len(ix) >= 3]
    genuine, impostor = [], []
    for n in people:
        ix = by[n]
        enrol, probes = ix[:2], ix[2:2 + per_person]
        genuine += [(enrol, p) for p in probes]
        others = rng.sample(people, impostors + 1)
        impostor += [(enrol, by[o][2]) for o in others if o != n][:impostors]
    return genuine, impostor


def rates(g, i, higher_is_same):
    """FRR at FAR 1e-3 and 1e-4, and the equal error rate."""
    g, i = np.asarray(g), np.asarray(i)
    if not higher_is_same:
        g, i = -g, -i
    out = {}
    for far in (1e-3, 1e-4):
        thr = np.quantile(i, 1 - far)
        out[far] = (float(np.mean(g <= thr)), float(thr))
    ts = np.quantile(np.concatenate([g, i]), np.linspace(0, 1, 2001))
    eer = min(ts, key=lambda t: abs(np.mean(g <= t) - np.mean(i > t)))
    out["eer"] = float((np.mean(g <= eer) + np.mean(i > eer)) / 2)
    return out


def main():
    names, V = embeddings()
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    rng = random.Random(0)
    genuine, impostor = pairs(names, rng)
    print(f"genuine pairs {len(genuine)}, impostor pairs {len(impostor)}\n")
    mean = V.mean(axis=0)
    P512 = np.random.default_rng(1).standard_normal((128, 512)).astype(np.float32)
    P1024 = np.random.default_rng(2).standard_normal((128, 1024)).astype(np.float32)

    def cosine(enrol, p, multi=False):
        e = V[enrol].mean(axis=0) if multi else V[enrol[0]]
        return float(e @ V[p] / np.linalg.norm(e))

    def ber_sign(enrol, p):
        return float(np.mean(((V[enrol[0]] - mean) > 0) != ((V[p] - mean) > 0)))

    def ber_proj(enrol, p):
        return float(np.mean((V[enrol[0]] @ P512 > 0) != (V[p] @ P512 > 0)))

    def ber_reliable(enrol, p, multi=False, keep=256):
        e = (V[enrol].mean(axis=0) if multi else V[enrol[0]]) @ P1024
        idx = np.argsort(-np.abs(e))[:keep]          # helper data: which bits
        return float(np.mean((e[idx] > 0) != ((V[p] @ P1024)[idx] > 0)))

    methods = [("cosine", lambda e, p: cosine(e, p), True),
               ("sign128", ber_sign, False), ("proj512", ber_proj, False),
               ("reliable", ber_reliable, False),
               ("multi", lambda e, p: ber_reliable(e, p, multi=True), False)]
    print(f"{'method':9} {'genuine: median  p90':>21} {'impostor: p1  median':>22} "
          f"{'FRR@FAR1e-3':>12} {'FRR@FAR1e-4':>12} {'EER':>6}  threshold@1e-4")
    for name, f, higher in methods:
        g = [f(e, p) for e, p in genuine]
        i = [f(e, p) for e, p in impostor]
        r = rates(g, i, higher)
        thr = r[1e-4][1] if higher else -r[1e-4][1]
        print(f"{name:9} {np.median(g):>12.3f} {np.quantile(g, 0.9 if not higher else 0.1):>7.3f} "
              f"{np.quantile(i, 0.01 if not higher else 0.99):>12.3f} {np.median(i):>8.3f} "
              f"{r[1e-3][0]:>12.1%} {r[1e-4][0]:>12.1%} {r['eer']:>6.1%}  {thr:.3f}")


if __name__ == "__main__":
    main()
