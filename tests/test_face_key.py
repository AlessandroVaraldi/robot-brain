#!/usr/bin/env python3
"""Faces as keys (pi/face_key): locks, the software chip, the store, the
admin. Faces here are random vectors and photos of them are noisy copies. No
models, no camera."""

import contextlib
import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi"))
from cryptography.exceptions import InvalidTag  # noqa: E402
from face_key import admin  # noqa: E402
from face_key.chip import ChipBusy, SoftChip  # noqa: E402
from face_key.lock import CODE_BITS, DIRS, KEY_BYTES, Lock, make_lock, project, try_key  # noqa: E402
from face_key.store import (ChipMismatch, FaceKeys, admin_keypair, listing,  # noqa: E402
                            open_as_admin)

RESULTS = []
rng = np.random.default_rng(0)


def check(ok, label):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


def face():
    v = rng.standard_normal(128)
    return v / np.linalg.norm(v)


def photo(f, noise=0.8):
    v = f + noise * rng.standard_normal(128) / np.sqrt(128)
    return v / np.linalg.norm(v)


def two(f):
    return [photo(f), photo(f)]


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


class Counting:
    """A chip that remembers what it was asked."""
    def __init__(self, chip):
        self.chip, self.asked = chip, []

    def mac(self, label, data=b""):
        self.asked.append(label)
        return self.chip.mac(label, data)


def locks():
    check(CODE_BITS == 510 and KEY_BYTES == 15, "510 face bits, a 120-bit key")
    check(hashlib.sha256(DIRS.tobytes()).hexdigest()[:16] == "a5383f0a29ebf4b9",
          "the projection is the same on every machine")
    f = face()
    lock, key = make_lock([photo(f), photo(f)])
    got = try_key(lock, project([photo(f)]))
    check(got and got[0] == key, f"another photo of the same face opens it ({got and got[1]} bits corrected)")
    again = Lock.from_bytes(lock.to_bytes())
    check(try_key(again, project([photo(f)]))[0] == key, "a lock stored and read back still opens")
    check(not any(try_key(lock, project([face()])) for _ in range(300)), "300 other faces do not")
    check(key not in lock.to_bytes(), "the key is not in the lock")

    lock, key = make_lock([f, f])
    for flips, opens in ((56, True), (57, False)):
        p = project([f])
        p[lock.kept[:flips]] *= -1
        got = try_key(lock, p)
        check(bool(got) == opens and (not got or got == (key, 56)),
              f"{flips} face bits changed: {'opens' if opens else 'stays shut'}")
    lock2, key2 = make_lock([photo(f), photo(f)], key)
    check(key2 == key and try_key(lock2, project([photo(f)]))[0] == key,
          "a new lock can hold the same key")


def chip(tmp):
    clock = Clock()
    c = SoftChip(tmp / "chip.key", per_minute=60, burst=3, clock=clock)
    check(oct((tmp / "chip.key").stat().st_mode & 0o777) == "0o600", "the soft chip's key file is private")
    a = c.mac(b"check", b"x")
    check(SoftChip(tmp / "chip.key").mac(b"check", b"x") == a, "same key file, same answers")
    check(SoftChip(key=os.urandom(32)).mac(b"check", b"x") != a, "another chip, other answers")
    check(c.mac(b"memory", b"x") != a, "the label is part of the answer")
    try:
        c.mac(b"anything")
        check(False, "an unknown label is refused")
    except ValueError:
        check(True, "an unknown label is refused")
    c.mac(b"check")                      # the third of three
    try:
        c.mac(b"check")
        check(False, "asked too often: busy")
    except ChipBusy:
        check(True, "asked too often: busy")
    clock.t += 1.0
    check(c.mac(b"check") is not None, "a second later it answers again")


def store(tmp):
    clock = Clock()
    private, public = admin_keypair()
    path = tmp / "faces.sqlite"
    soft = SoftChip(key=os.urandom(32), burst=1000)
    chip_ = Counting(soft)
    keys = FaceKeys(chip_, public, path, clock=clock)

    luca, anna = face(), face()
    p = keys.enrol(two(luca), "Luca")
    check(not p.known and p.name == "Luca", "enrolled: new")
    got = keys.recognise(two(luca))
    check(got and got.id == p.id and got.name == "Luca" and got.known, "recognised from two other frames")
    check(keys.recognise([photo(luca), photo(face())]) is None,
          "one frame of them and one of someone else: nobody")
    try:
        keys.recognise([photo(luca)])
        check(False, "one frame is not enough to recognise")
    except ValueError:
        check(True, "one frame is not enough to recognise")
    check(got.memory_key == p.memory_key, "the same memory key every time")
    q = keys.enrol(two(anna), "Anna")
    check(keys.recognise(two(anna)).id == q.id and keys.recognise(two(luca)).id == p.id,
          "two people, each recognised as themselves")
    check(q.memory_key != p.memory_key, "each their own memory key")

    chip_.asked.clear()
    strangers = [keys.recognise(two(face())) for _ in range(100)]
    check(not any(strangers), "100 strangers: nobody")
    check(chip_.asked == [], "and the chip was not even asked")

    again = keys.enrol(two(luca), "Luca")
    check(again.known and again.id == p.id and len(keys) == 2, "the same face enrolled twice: once")

    raw = path.read_bytes()
    check(b"Luca" not in raw and b"Anna" not in raw, "no name in the file")

    reopened = FaceKeys(soft, public, path, clock=clock)
    got = reopened.recognise(two(luca))
    check(got and got.id == p.id and got.memory_key == p.memory_key, "still recognised after a restart")
    try:
        FaceKeys(SoftChip(key=os.urandom(32)), public, path)
        check(False, "another chip cannot open the stored locks")
    except ChipMismatch:
        check(True, "another chip cannot open the stored locks")

    def lock_count(pid):
        return sum(1 for r in keys._locks if r[1] == pid)

    keys.recognise(two(luca))
    check(lock_count(p.id) == 1, "recognised soon after: no new lock")
    clock.t += 86400
    keys.recognise(two(luca))
    check(lock_count(p.id) == 2, "a day later: a new lock")
    keys.recognise(two(luca))
    check(lock_count(p.id) == 2, "and only one")
    for _ in range(3):
        clock.t += 86400
        keys.recognise(two(luca))
    stored = {pid: locks for pid, _, _, locks in listing(path)}
    check(lock_count(p.id) == 3 and stored[p.id] == 3, "never more than three locks")
    check(FaceKeys(soft, public, path, clock=clock).recognise(two(luca)).id == p.id,
          "the newest locks open after a restart")

    check(keys.knows_name("LUCA") and keys.knows_name(" luca ") and not keys.knows_name("Marco")
          and not keys.knows_name(""), "John can tell it knows a Luca, not a Marco")
    keys.rename(q.id, "Hanna")
    check(FaceKeys(soft, public, path).recognise(two(anna)).name == "Hanna",
          "a name corrected is the name from then on")
    check(keys.knows_name("Hanna") and not keys.knows_name("Anna"), "and the name tags follow it")
    r = keys.enrol(two(face()), "Anna Maria")
    check(all(keys.knows_name(n) for n in ("anna maria", "Anna", "MARIA")),
          "a name of two words: known whole and by each word")
    keys.forget(r.id)
    check(not keys.knows_name("Maria"), "forgotten: the name too")

    names = sorted(open_as_admin(private, sealed) for _, sealed, _, _ in listing(path))
    check(names == ["Hanna", "Luca"], "the admin reads the names with the private key")
    other, _ = admin_keypair()
    try:
        open_as_admin(other, listing(path)[0][1])
        check(False, "not with another key")
    except InvalidTag:
        check(True, "not with another key")

    key_file = tmp / "admin.key"
    key_file.write_text(private.hex())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        admin.main(["list", "--key", str(key_file), "--store", str(path)])
        admin.main(["delete", "luca", "--key", str(key_file), "--store", str(path)])
    check("Luca" in out.getvalue() and "would delete: Luca" in out.getvalue() and len(keys) == 2,
          "admin: list, and delete shows who would go")
    with contextlib.redirect_stdout(io.StringIO()):
        admin.main(["delete", "LUCA", "--key", str(key_file), "--store", str(path), "--yes"])
    check(len(keys) == 1 and keys.recognise(two(luca)) is None and not keys.knows_name("Luca"),
          "deleted by name: gone, even for the robot already running")
    check(keys.recognise(two(anna)).name == "Hanna", "the others stay")

    try:
        keys.forget(p.id)
        check(False, "only a face just recognised can be forgotten")
    except KeyError:
        check(True, "only a face just recognised can be forgotten")
    keys.forget(q.id)
    check(len(keys) == 0 and keys.recognise(two(anna)) is None, "forget my face: forgotten")

    try:
        keys.enrol([photo(anna)], "Anna")
        check(False, "enrolling needs two frames")
    except ValueError:
        check(True, "enrolling needs two frames")

    with contextlib.redirect_stdout(io.StringIO()) as printed:
        admin.main(["keygen", str(tmp / "new.key")])
    pub = bytes.fromhex(printed.getvalue().strip())
    priv = bytes.fromhex((tmp / "new.key").read_text())
    from face_key.store import seal_for_admin
    check(open_as_admin(priv, seal_for_admin(pub, "Zoe")) == "Zoe"
          and oct((tmp / "new.key").stat().st_mode & 0o777) == "0o600",
          "keygen: a working pair, the private file private")


def escrow(tmp):
    import json
    soft = SoftChip(key=os.urandom(32), burst=1000)
    file = tmp / "debug" / "keys.json"
    keys = FaceKeys(soft, admin_keypair()[1], tmp / "escrowed.sqlite", escrow=file)
    zoe = face()
    p = keys.enrol(two(zoe), "Zoe")
    kept = json.loads(file.read_text())
    check(kept[p.id] == {"name": "Zoe", "memory_key": p.memory_key.hex()}
          and kept["_store"]["locks_key"] == soft.mac(b"locks").hex()
          and oct(file.stat().st_mode & 0o777) == "0o600", "debug escrow: every key in the clear, file private")
    keys.rename(p.id, "Zoey")
    check(json.loads(file.read_text())[p.id]["name"] == "Zoey", "escrow: a new name")
    keys.forget(p.id)
    check(p.id not in json.loads(file.read_text()), "escrow: forgotten, gone from it too")
    plain = FaceKeys(soft, admin_keypair()[1], tmp / "plain.sqlite")
    plain.enrol(two(face()), "Ugo")
    check(not (tmp / "keys.json").exists() and plain.escrow is None, "no escrow unless asked")


def main():
    with tempfile.TemporaryDirectory() as t:
        locks()
        chip(Path(t))
        store(Path(t))
        escrow(Path(t))
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
