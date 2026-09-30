"""A face turned into a key and back (a fuzzy commitment).

A face vector (SFace, 128 numbers) is projected on 4096 fixed random
directions. At enrolment the 510 projections furthest from zero, the ones least
likely to change sign on another photo, are kept, and their signs are the face
bits. A random 120-bit key is encoded with a BCH code (511 bits, corrects 56
errors) and XORed with the face bits: that, with which projections were kept,
is the lock. Neither the face nor the key is in it.

Another photo of the same person gives bits that differ in a few places; the
code corrects up to 56 of them and gives the key back. A different person's
bits differ in about half of them: on LFW, of 429,680 pairs of different
people, the nearest differed in 62 (eval/face_bits.py).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import bchlib
import numpy as np

DIM = 128                 # an SFace vector
DIRECTIONS = 4096         # fixed random projections
_BCH = bchlib.BCH(56, m=9)                        # 511 bits, corrects 56
KEY_BYTES = (_BCH.n - _BCH.ecc_bits) // 8         # 15: a 120-bit key
KEY_BITS = 8 * KEY_BYTES
CODE_BITS = KEY_BITS + _BCH.ecc_bits              # 510: the face bits used
_PAD = 8 * _BCH.ecc_bytes - _BCH.ecc_bits         # bchlib's ecc is whole bytes


def _directions() -> np.ndarray:
    """±1 entries from SHAKE-256: the same projection on every machine and
    with every numpy, or the locks made on one would not open on another."""
    raw = hashlib.shake_256(b"john/face_key/directions/v1").digest(DIM * DIRECTIONS // 8)
    bits = np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(DIM, DIRECTIONS)
    return bits.astype(np.float32) * 2 - 1


DIRS = _directions()


def project(vecs) -> np.ndarray:
    """One or more face vectors of the same person (each made unit length,
    then averaged) on the fixed directions."""
    units = [np.asarray(v, np.float32).ravel() for v in vecs]
    return np.mean([u / np.linalg.norm(u) for u in units], axis=0) @ DIRS


@dataclass
class Lock:
    kept: np.ndarray      # which projections, most reliable first (CODE_BITS indices)
    helper: np.ndarray    # codeword XOR face bits (CODE_BITS bits)

    def to_bytes(self) -> bytes:
        return self.kept.astype("<u2").tobytes() + np.packbits(self.helper).tobytes()

    @classmethod
    def from_bytes(cls, b: bytes) -> "Lock":
        n = 2 * CODE_BITS
        kept = np.frombuffer(b[:n], "<u2").astype(np.int64)
        helper = np.unpackbits(np.frombuffer(b[n:], np.uint8))[:CODE_BITS]
        return cls(kept, helper)


def _codeword(key: bytes) -> np.ndarray:
    ecc = np.unpackbits(np.frombuffer(bytes(_BCH.encode(key)), np.uint8))[:_BCH.ecc_bits]
    return np.concatenate([np.unpackbits(np.frombuffer(key, np.uint8)), ecc])


def make_lock(vecs, key: bytes | None = None) -> tuple[Lock, bytes]:
    """A lock for the face in `vecs` (frontal vectors of one person; two or
    more make a steadier lock) holding `key`, or a new random key."""
    key = os.urandom(KEY_BYTES) if key is None else key
    assert len(key) == KEY_BYTES
    p = project(vecs)
    kept = np.argsort(-np.abs(p), kind="stable")[:CODE_BITS]
    return Lock(kept, _codeword(key) ^ (p[kept] > 0).astype(np.uint8)), key


def try_key(lock: Lock, p: np.ndarray) -> tuple[bytes, int] | None:
    """The key in `lock` and how many bits had to be corrected, if the face
    projected in `p` is close enough to the one that closed it; else None."""
    x = lock.helper ^ (p[lock.kept] > 0).astype(np.uint8)
    key = bytearray(np.packbits(x[:KEY_BITS]).tobytes())
    ecc = bytearray(np.packbits(np.concatenate([x[KEY_BITS:], np.zeros(_PAD, np.uint8)])).tobytes())
    errors = _BCH.decode(key, ecc)
    if errors < 0:
        return None
    _BCH.correct(key, ecc)
    return bytes(key), errors
