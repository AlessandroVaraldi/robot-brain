#!/usr/bin/env python3
"""Everything the robot keeps, decrypted: for a debug memory (one_mind.py
--debug), whose keys were written in the clear to keys.json next to it.

Per person: the name, the face id, the locks (when made, and whether they open
with the key on file), the notes and the episodes.  Without keys.json a person
still shows, as locked; with the admin key their name shows too.

    python3 john/inspect_memory.py                     # memory/debug
    python3 john/inspect_memory.py memory/debug --admin-key memory/debug/admin.key
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from memory_store import MemoryStore
from paths import DEBUG_MEMORY, FACE_KEYS

sys.path.insert(0, str(FACE_KEYS))
from face_key import store as face_store  # noqa: E402


def day(t: float) -> str:
    return time.strftime("%d/%m/%Y %H:%M", time.localtime(t))


def inspect(where: Path, admin_key: bytes | None = None) -> str:
    keys_file = where / "keys.json"
    keys = json.loads(keys_file.read_text()) if keys_file.exists() else {}
    locks_key = bytes.fromhex(keys.get("_store", {}).get("locks_key", "")) or None
    memory = MemoryStore(where / "memory.sqlite")
    faces = sqlite3.connect(str(where / "faces.sqlite"))
    faces.executescript(face_store.SCHEMA)
    out = [f"{where}: {'keys on file' if keys else 'no keys.json: contents stay locked'}"]

    people = faces.execute("SELECT id, sealed_name, created FROM people ORDER BY created").fetchall()
    in_memory = {r["face_id"]: r for r in memory.db.execute("SELECT * FROM persons")}
    for face_id, sealed, created in people:
        kept = keys.get(face_id, {})
        name = kept.get("name") or (face_store.open_as_admin(admin_key, sealed) if admin_key else "?")
        out.append(f"\n{name}  (face {face_id}, since {day(created)})")
        for lock_id, box, made in faces.execute(
                "SELECT id, box, created FROM locks WHERE person = ? ORDER BY id", (face_id,)):
            opens = "?"
            if locks_key:
                try:
                    face_store._unseal(locks_key, box, face_id.encode())
                    opens = "opens"
                except Exception:
                    opens = "does NOT open with the key on file"
            out.append(f"  lock {lock_id}: made {day(made)}, {opens}")
        tags = faces.execute("SELECT COUNT(*) FROM name_tags WHERE person = ?", (face_id,)).fetchone()[0]
        out.append(f"  name tags: {tags}")
        row = in_memory.pop(face_id, None)
        if row is None:
            out.append("  memory: nothing yet")
            continue
        if "memory_key" not in kept:
            out.append("  memory: locked (no key on file)")
            continue
        key = bytes.fromhex(kept["memory_key"])
        out.append(f"  notes: {memory.notes(row['id'], key) or '(none)'}")
        for e in reversed(memory.recent_episodes(row["id"], key, 1000)):
            out.append(f"  episode {day(e['started_at'])}: {e['summary']}")
    for face_id in in_memory:
        out.append(f"\n(memory for face {face_id}, which has no locks: an orphan)")
    facts = memory.self_facts()
    if facts:
        out.append("\nwhat the lab told the robot:")
        out += [f"  #{f['id']} {day(f['created_at'])}: {f['text']}" for f in facts]
    out.append(f"\n{len(people)} people")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("where", nargs="?", type=Path, default=DEBUG_MEMORY)
    ap.add_argument("--admin-key", type=Path, help="the admin's private key file, for names")
    a = ap.parse_args()
    admin = bytes.fromhex(a.admin_key.read_text().strip()) if a.admin_key else None
    print(inspect(a.where, admin))


if __name__ == "__main__":
    main()
