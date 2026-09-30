#!/usr/bin/env python3
"""The debug memory: a test admin key made on the spot, every key written in
the clear, and inspect_memory.py reading back all that is kept encrypted.
No models, no camera."""

import io
import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import one_mind  # noqa: E402
from inspect_memory import inspect  # noqa: E402
from memory_store import MemoryStore  # noqa: E402

RESULTS = []
rng = np.random.default_rng(0)


def check(ok, label):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


def frames(face):
    return [face + 0.8 * rng.standard_normal(128) / np.sqrt(128) for _ in range(2)]


def main():
    with tempfile.TemporaryDirectory() as t:
        where = Path(t)
        with contextlib.redirect_stdout(io.StringIO()):
            none = one_mind.face_keys("", where)
            keys = one_mind.face_keys("", where, debug=True)
        check(none is None, "not debug, no admin key: nobody remembered")
        check(keys is not None and (where / "admin.pub").exists()
              and oct((where / "admin.key").stat().st_mode & 0o777) == "0o600",
              "debug: a test admin key made, the private one private")

        alessandro, maria = rng.standard_normal(128), rng.standard_normal(128)
        memory = MemoryStore(where / "memory.sqlite")
        v = keys.enrol(frames(alessandro), "Alessandro")
        pid, _ = memory.person(v.id)
        memory.update_notes(pid, "Alessandro is building a robot arm.", v.memory_key)
        memory.add_episode(pid, 1_000_000, 1_000_100, "Alessandro introduced himself.", v.memory_key)
        keys.enrol(frames(maria), "Maria")
        memory.add_self_fact("Your hand was replaced.")

        seen = inspect(where)
        check(all(x in seen for x in ("Alessandro  (face", "Maria  (face", "opens",
                                      "notes: Alessandro is building a robot arm.",
                                      "Alessandro introduced himself.", "memory: nothing yet",
                                      "Your hand was replaced.", "2 people")),
              "inspect: names, locks, notes, episodes, the lab's facts")

        (where / "keys.json").rename(where / "keys.away")
        locked = inspect(where)
        check("?  (face" in locked and "memory: locked" in locked and "robot arm" not in locked,
              "without the keys: nothing readable")
        admin = bytes.fromhex((where / "admin.key").read_text().strip())
        named = inspect(where, admin)
        check("Alessandro  (face" in named and "robot arm" not in named,
              "with the admin key: the names, and still not the notes")
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
