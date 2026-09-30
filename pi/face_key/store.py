"""What John keeps about faces, and nothing that shows one.

Only people who told John their name and agreed to have their face remembered
are here. Each has a random id and:

  locks        up to three (lock.py), stored encrypted under a key from the
               chip: without the chip a photo cannot even be tried against them
  profile      their name, encrypted under their memory key, which only their
               face and the chip together give
  sealed name  their name again, sealed with the admin's public key: the robot
               can write it and never read it; the admin, whose private key is
               kept off the robot, can list and delete people by name (admin.py)
  name tags    keyed hashes of the name and of each word of it, under a key
               from the chip: John can tell whether it knows someone called
               Alessandro without keeping a list of names

A person is who their face opens, not what they are called: the name is
something John knows about that face, and can be corrected.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .lock import Lock, make_lock, project, try_key

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "memory" / "faces.sqlite"
ADMIN_PUB = ROOT / "memory" / "admin.pub"          # hex, from `admin.py keygen`
SOFT_CHIP_KEY = ROOT / "memory" / "soft_chip.key"  # development only

SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY, profile BLOB NOT NULL, sealed_name BLOB NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS locks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, person TEXT NOT NULL, box BLOB NOT NULL,
    created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS name_tags (person TEXT NOT NULL, tag BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS name_tags_by_tag ON name_tags(tag);
"""


def name_parts(name: str) -> set[str]:
    """A name, and each word of it, as they are compared: case and spacing
    do not matter."""
    whole = " ".join(name.split()).casefold()
    return {whole, *whole.split()} - {""}


class ChipMismatch(Exception):
    """The stored locks were made with another chip."""


@dataclass
class Person:
    id: str
    name: str
    memory_key: bytes     # for what the brain keeps about them
    known: bool = True    # False only when enrol() has just added them
    errors: int = 0       # most face bits corrected in one frame (fewer: closer)


def _seal(key: bytes, plain: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plain, aad)


def _unseal(key: bytes, blob: bytes, aad: bytes) -> bytes:
    return AESGCM(key).decrypt(blob[:12], blob[12:], aad)


def _raw(public) -> bytes:
    return public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _admin_box_key(shared: bytes, ephemeral: bytes, admin: bytes) -> bytes:
    return HKDF(hashes.SHA256(), 32, None, b"john/face_key/name" + ephemeral + admin).derive(shared)


def admin_keypair() -> tuple[bytes, bytes]:
    """A new (private, public) pair for the admin, 32 bytes each."""
    private = X25519PrivateKey.generate()
    raw = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                serialization.NoEncryption())
    return raw, _raw(private.public_key())


def seal_for_admin(admin_public: bytes, name: str) -> bytes:
    ephemeral = X25519PrivateKey.generate()
    e = _raw(ephemeral.public_key())
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(admin_public))
    return e + _seal(_admin_box_key(shared, e, admin_public), name.encode(), b"")


def open_as_admin(admin_private: bytes, sealed: bytes) -> str:
    private = X25519PrivateKey.from_private_bytes(admin_private)
    e, rest = sealed[:32], sealed[32:]
    shared = private.exchange(X25519PublicKey.from_public_bytes(e))
    return _unseal(_admin_box_key(shared, e, _raw(private.public_key())), rest, b"").decode()


class FaceKeys:
    MAX_LOCKS = 3                  # per person: the newest are kept
    REFRESH_AFTER_S = 20 * 3600    # a new lock at most this often

    def __init__(self, chip, admin_public: bytes, path: Path = STORE, clock=time.time,
                 escrow: Path | None = None):
        """`escrow`: debug only.  A file where every key is written in the
        clear, so that everything stored can be read (john/inspect_memory.py):
        it undoes all the protection above."""
        if len(admin_public) != 32:
            raise ValueError("the admin public key is 32 bytes")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.chip, self.clock, self.admin_public = chip, clock, admin_public
        self._box = chip.mac(b"locks")
        self._names = chip.mac(b"names")
        self.escrow = Path(escrow) if escrow else None
        self._escrow("_store", locks_key=self._box.hex(), names_key=self._names.hex())
        self._locks = [self._unbox(*row) for row in
                       self.db.execute("SELECT id, person, box, created FROM locks ORDER BY id")]
        self._open = {}            # id -> (key, memory key), for faces seen since start

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM people").fetchone()[0]

    def _unbox(self, lock_id, person, box, created):
        try:
            raw = _unseal(self._box, box, person.encode())
        except InvalidTag:
            raise ChipMismatch("these locks were not made with this chip") from None
        return lock_id, person, Lock.from_bytes(raw[:-32]), raw[-32:], created

    def _add_lock(self, person: str, vecs, key: bytes):
        lock, _ = make_lock(vecs, key)
        check, now = self.chip.mac(b"check", key), self.clock()
        box = _seal(self._box, lock.to_bytes() + check, person.encode())
        cur = self.db.execute("INSERT INTO locks (person, box, created) VALUES (?, ?, ?)",
                              (person, box, now))
        self._locks.append((cur.lastrowid, person, lock, check, now))
        theirs = sorted(r[0] for r in self._locks if r[1] == person)
        old = set(theirs[:-self.MAX_LOCKS])
        self.db.executemany("DELETE FROM locks WHERE id = ?", [(i,) for i in old])
        self._locks = [r for r in self._locks if r[0] not in old]
        self.db.commit()

    def _tag(self, part: str) -> bytes:
        return hmac.new(self._names, part.encode(), "sha256").digest()

    def _set_name(self, person: str, name: str, memory_key: bytes):
        self.db.execute("UPDATE people SET profile = ?, sealed_name = ? WHERE id = ?", (
            _seal(memory_key, json.dumps({"name": name}).encode(), person.encode()),
            seal_for_admin(self.admin_public, name), person))
        self.db.execute("DELETE FROM name_tags WHERE person = ?", (person,))
        self.db.executemany("INSERT INTO name_tags VALUES (?, ?)",
                            [(person, self._tag(p)) for p in name_parts(name)])
        self.db.commit()

    def knows_name(self, name: str) -> bool:
        """Whether someone remembered is called this (or has it as one word of
        their name)."""
        whole = " ".join(name.split()).casefold()
        return bool(whole) and self.db.execute(
            "SELECT 1 FROM name_tags WHERE tag = ? LIMIT 1", (self._tag(whole),)).fetchone() is not None

    def _escrow(self, person: str, **fields):
        if self.escrow is None:
            return
        kept = json.loads(self.escrow.read_text()) if self.escrow.exists() else {}
        if fields:
            kept[person] = {**kept.get(person, {}), **fields}
        else:
            kept.pop(person, None)
        self.escrow.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.escrow, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, json.dumps(kept, indent=1).encode())
        os.close(fd)

    def _drop(self, person: str):
        self.db.execute("DELETE FROM name_tags WHERE person = ?", (person,))
        self.db.execute("DELETE FROM locks WHERE person = ?", (person,))
        self.db.execute("DELETE FROM people WHERE id = ?", (person,))
        self.db.commit()
        self._locks = [r for r in self._locks if r[1] != person]
        self._open.pop(person, None)
        self._escrow(person)

    def _opened(self, frame) -> dict:
        """person -> (key, bits corrected, check) for each person one of whose
        locks this single frame opens."""
        p, out = project([frame]), {}
        for _, person, lock, check, _ in self._locks:
            got = try_key(lock, p)
            if got and (person not in out or got[1] < out[person][1]):
                out[person] = (got[0], got[1], check)
        return out

    def recognise(self, frames) -> Person | None:
        """Whose face this is, from two frontal frames of it or more, or None.

        Each frame must open a lock of the same person on its own: one frame
        of a close lookalike can open a lock (on LFW, a father his daughter's),
        two rarely both do. A person whose newest lock is old gets a new one
        from these frames, so that a face that slowly changes is still
        recognised."""
        if len(frames) < 2:
            raise ValueError("recognise from two frontal frames or more")
        both = self._opened(frames[0])
        for frame in frames[1:]:
            here = self._opened(frame)
            both = {person: (key, max(errors, here[person][1]), check)
                    for person, (key, errors, check) in both.items()
                    if person in here and here[person][0] == key}
        for person, (key, errors, check) in sorted(both.items(), key=lambda kv: kv[1][1]):
            if not hmac.compare_digest(self.chip.mac(b"check", key), check):
                continue
            row = self.db.execute("SELECT profile FROM people WHERE id = ?", (person,)).fetchone()
            if row is None:                   # deleted by the admin while running
                self._drop(person)
                continue
            memory_key = self.chip.mac(b"memory", key)
            self._open[person] = (key, memory_key)
            name = json.loads(_unseal(memory_key, row[0], person.encode()))["name"]
            self._escrow(person, name=name, memory_key=memory_key.hex())
            newest = max(r[4] for r in self._locks if r[1] == person)
            if self.clock() - newest > self.REFRESH_AFTER_S:
                self._add_lock(person, frames, key)
            return Person(person, name, memory_key, errors=errors)
        return None

    def enrol(self, frames, name: str) -> Person:
        """Remember this face (two frontal frames of it or more), with this
        name: only for someone who has agreed. A face already known is not
        added twice: that person is returned."""
        if len(frames) < 2:
            raise ValueError("enrol from two frontal frames or more")
        found = self.recognise(frames)
        if found:
            return found
        person = secrets.token_hex(8)
        lock, key = make_lock(frames)
        memory_key = self.chip.mac(b"memory", key)
        self.db.execute("INSERT INTO people VALUES (?, ?, ?, ?)", (person, b"", b"", self.clock()))
        self._set_name(person, name, memory_key)
        self._add_lock(person, frames, key)
        self._open[person] = (key, memory_key)
        self._escrow(person, name=name, memory_key=memory_key.hex())
        return Person(person, name, memory_key, known=False)

    def rename(self, person: str, name: str):
        """Correct the name of someone recognised since start."""
        self._set_name(person, name, self._open[person][1])
        self._escrow(person, name=name)

    def forget(self, person: str):
        """Forget this face and its name: for someone recognised who asked."""
        if person not in self._open:
            raise KeyError("only a face recognised since start can be forgotten this way")
        self._drop(person)


# ---- for the admin: no chip, no faces, names only with the private key

def listing(path: Path = STORE) -> list[tuple[str, bytes, float, int]]:
    """(id, sealed name, created, locks) for everyone stored."""
    db = sqlite3.connect(str(path))
    return db.execute("SELECT p.id, p.sealed_name, p.created, COUNT(l.id) FROM people p "
                      "LEFT JOIN locks l ON l.person = p.id GROUP BY p.id "
                      "ORDER BY p.created").fetchall()


def delete(person: str, path: Path = STORE):
    db = sqlite3.connect(str(path))
    db.executescript(SCHEMA)
    db.execute("DELETE FROM name_tags WHERE person = ?", (person,))
    db.execute("DELETE FROM locks WHERE person = ?", (person,))
    db.execute("DELETE FROM people WHERE id = ?", (person,))
    db.commit()
