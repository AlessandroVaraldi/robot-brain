#!/usr/bin/env python3
"""Long-term memory for the robot: people and the encounters it had with them.

* The archive and the context are separate.  Everything is kept on disk; only
  a small, budgeted slice about the person in front of the robot goes into the
  prompt.
* A person is a face (pi/face_key), not a name: only people who introduced
  themselves and agreed to have their face remembered are here, under the id
  their face opens.  What they told the robot is encrypted under the memory
  key that only their face and the robot's chip give: read while they are in
  front of it, never otherwise.  Their name is not here at all.
* A fact is never removed by the absence of one: updates that come back empty
  leave the stored value alone.

sqlite3 and `cryptography`, written transactionally.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY,
    face_id     TEXT NOT NULL UNIQUE,      -- the id their face opens (face_key)
    notes       BLOB,                      -- what they have told the robot, encrypted
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES persons(id),
    started_at  REAL NOT NULL,
    ended_at    REAL NOT NULL,
    summary     BLOB NOT NULL              -- encrypted
);
CREATE INDEX IF NOT EXISTS episodes_by_person ON episodes(person_id, ended_at);
-- What the lab team has told the robot about itself.  Written only by an
-- operator (self_memory.py), never from the whiteboard.
CREATE TABLE IF NOT EXISTS self_facts (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


def name_key(name: str) -> str:
    """How two spellings of a name are recognised as the same name.

    Case and spacing only.  No guessing at variants ("Vale" for "Alessandro").
    """
    return " ".join(name.split()).casefold()


def approx_tokens(text: str) -> int:
    return len(text) // 4 + 1


def _seal(key: bytes, text: str, aad: str) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, text.encode(), aad.encode())


def _unseal(key: bytes, blob: bytes, aad: str) -> str:
    return AESGCM(key).decrypt(blob[:12], blob[12:], aad.encode()).decode()


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- people ------------------------------------------------------------
    def person(self, face_id: str) -> tuple[int, bool]:
        """(id, created) for the person whose face opens `face_id`."""
        row = self.db.execute("SELECT id FROM persons WHERE face_id = ?", (face_id,)).fetchone()
        if row is not None:
            return row["id"], False
        now = time.time()
        with self.db:
            cur = self.db.execute("INSERT INTO persons (face_id, created_at, updated_at) "
                                  "VALUES (?, ?, ?)", (face_id, now, now))
        return cur.lastrowid, True

    def people(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    def forget(self, face_id: str) -> bool:
        """Everything about this person, gone."""
        row = self.db.execute("SELECT id FROM persons WHERE face_id = ?", (face_id,)).fetchone()
        if row is None:
            return False
        with self.db:
            self.db.execute("DELETE FROM episodes WHERE person_id = ?", (row["id"],))
            self.db.execute("DELETE FROM persons WHERE id = ?", (row["id"],))
        return True

    def notes(self, person_id: int, key: bytes) -> str:
        row = self.db.execute("SELECT notes FROM persons WHERE id = ?", (person_id,)).fetchone()
        if row is None or not row["notes"]:
            return ""
        return _unseal(key, row["notes"], f"notes:{person_id}")

    def update_notes(self, person_id: int, notes: str, key: bytes):
        """Replace the notes - but an empty update never wipes what is known."""
        notes = (notes or "").strip()
        if not notes:
            return
        with self.db:
            self.db.execute("UPDATE persons SET notes = ?, updated_at = ? WHERE id = ?",
                            (_seal(key, notes, f"notes:{person_id}"), time.time(), person_id))

    def rename(self, person_id: int, old: str, new: str, key: bytes):
        """Their name changed: the old one, wherever their notes and episodes
        say it, becomes the new one - or "Alessandro is building a robot arm"
        is read about someone now called Luca."""
        old_word = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
        with self.db:
            notes = self.notes(person_id, key)
            if notes:
                self.db.execute("UPDATE persons SET notes = ? WHERE id = ?", (
                    _seal(key, old_word.sub(new, notes), f"notes:{person_id}"), person_id))
            for row in self.db.execute("SELECT id, summary FROM episodes WHERE person_id = ?",
                                       (person_id,)).fetchall():
                text = _unseal(key, row["summary"], f"episode:{person_id}")
                self.db.execute("UPDATE episodes SET summary = ? WHERE id = ?", (
                    _seal(key, old_word.sub(new, text), f"episode:{person_id}"), row["id"]))

    # -- encounters ----------------------------------------------------------
    def add_episode(self, person_id: int, started_at: float, ended_at: float, summary: str,
                    key: bytes):
        summary = (summary or "").strip()
        if not summary:
            return None
        with self.db:
            cur = self.db.execute(
                "INSERT INTO episodes (person_id, started_at, ended_at, summary) "
                "VALUES (?, ?, ?, ?)",
                (person_id, started_at, ended_at, _seal(key, summary, f"episode:{person_id}")))
        return cur.lastrowid

    def recent_episodes(self, person_id: int, key: bytes, n: int = 5) -> list[dict]:
        return [{"started_at": r["started_at"], "ended_at": r["ended_at"],
                 "summary": _unseal(key, r["summary"], f"episode:{person_id}")}
                for r in self.db.execute(
                    "SELECT * FROM episodes WHERE person_id = ? ORDER BY ended_at DESC LIMIT ?",
                    (person_id, n))]

    # -- what goes into the prompt -----------------------------------------------
    def context_for(self, person_id: int, name: str, key: bytes, budget_tokens: int = 300,
                    now: float | None = None) -> str:
        """The slice of memory about ONE person, within a token budget.

        Only this person's row is read, and only they can open it, so nothing
        about anyone else can reach the prompt.  It holds the distilled profile
        and when they last met, but not the encounter narratives: with a line
        like "Alessandro said he is building a robot arm" in the prompt, "WHAT AM I
        BUILDING?" was answered "I am building a robot arm".  Episodes are for
        building the profile, not for replying.
        """
        now = time.time() if now is None else now
        if self.db.execute("SELECT 1 FROM persons WHERE id = ?", (person_id,)).fetchone() is None:
            return ""
        last = self.db.execute("SELECT MAX(ended_at) FROM episodes WHERE person_id = ?",
                               (person_id,)).fetchone()[0]
        first = f"You have met {name} before" + (
            f", most recently {when(now - last)}." if last is not None else ".")
        notes = self.notes(person_id, key)
        if not notes:
            return first
        prefix = "What they have told you: "
        room = budget_tokens - approx_tokens(first) - approx_tokens(prefix)
        if room <= 0:
            return first
        if approx_tokens(notes) > room:
            notes = notes[:max(0, room * 4 - 4)].rsplit(" ", 1)[0] + "..."
        return first + "\n" + prefix + notes

    # -- the robot itself ---------------------------------------------------------
    def add_self_fact(self, text: str, now: float | None = None):
        text = " ".join((text or "").split())[:300]
        if not text:
            raise ValueError("empty fact")
        with self.db:
            cur = self.db.execute("INSERT INTO self_facts (text, created_at) VALUES (?, ?)",
                                  (text, time.time() if now is None else now))
        return cur.lastrowid

    def remove_self_fact(self, fact_id: int) -> bool:
        with self.db:
            cur = self.db.execute("DELETE FROM self_facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0

    def self_facts(self):
        return self.db.execute(
            "SELECT * FROM self_facts ORDER BY created_at DESC, id DESC").fetchall()

    def self_context(self, now: float | None = None, budget_tokens: int = 150,
                     started_at: float | None = None) -> str:
        """What the robot knows about itself, within a token budget.

        Its experience is counted, never narrated: how many people, since when,
        and no names, so nothing about who else came by can reach the person in
        front of it.  Counts are of encounters already over, hence "before this
        conversation": the current person has not been archived yet.
        """
        now = time.time() if now is None else now
        today = self.db.execute(
            "SELECT COUNT(DISTINCT person_id) FROM episodes WHERE ended_at >= ?",
            (start_of_day(now),)).fetchone()[0]
        # No line when the count is zero: "nobody has told you their name today"
        # contradicts someone who just has, and "WHAT IS MY NAME?" was then
        # sometimes answered "You are John".
        lines = [f"Before this conversation, {today} "
                 f"{'person has' if today == 1 else 'people have'} told you their name today."
                 ] if today else []
        known, since = self.db.execute(
            "SELECT COUNT(*), MIN(created_at) FROM persons").fetchone()
        if known:
            lines.append(f"Altogether you know {known} "
                         f"{'person' if known == 1 else 'people'} by name, the first "
                         f"met {when(now - since)}.")
        if started_at is not None:
            lines.append(f"You were switched on {when(now - started_at)}.")
        header = "The lab team has told you:"
        used = sum(approx_tokens(x) for x in lines) + approx_tokens(header)
        told = []
        for f in self.self_facts():
            line = f"- {f['text']} ({when(now - f['created_at'])})"
            if used + approx_tokens(line) > budget_tokens:
                break
            told.append(line)
            used += approx_tokens(line)
        if told:
            lines.append(header)
            lines.extend(told)
        return "\n".join(lines)

    def close(self):
        self.db.close()


def start_of_day(now: float) -> float:
    t = time.localtime(now)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def when(age_s: float) -> str:
    """Relative time, the way the robot should talk about it."""
    if age_s < 3600:
        return "earlier today" if age_s > 600 else "a few minutes ago"
    days = int(age_s // 86400)
    if days == 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        return f"{days // 7} week{'s' if days >= 14 else ''} ago"
    return f"{days // 30} month{'s' if days >= 60 else ''} ago"
