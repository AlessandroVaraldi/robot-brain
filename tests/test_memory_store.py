#!/usr/bin/env python3
"""Tests for memory_store.  No models, no robot, deterministic."""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
from cryptography.exceptions import InvalidTag  # noqa: E402
from memory_store import MemoryStore, approx_tokens, name_key, start_of_day, when  # noqa: E402

KV, KM = os.urandom(32), os.urandom(32)      # Alessandro's and Maria's memory keys

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def fresh():
    return MemoryStore(Path(tempfile.mkdtemp()) / "mem.sqlite")


@case
def test_spellings_of_a_name_are_one_name():
    assert name_key("ALESSANDRO") == name_key(" alessandro ") == name_key("Alessandro")
    assert name_key("Alessandro Bianchi") != name_key("Alessandro")
    return "ALESSANDRO / alessandro / ' Alessandro ' -> same key; a full name stays distinct"


@case
def test_a_face_is_one_person():
    m = fresh()
    a, created_a = m.person("face-1")
    b, created_b = m.person("face-1")
    c, _ = m.person("face-2")
    assert created_a and not created_b and a == b != c and m.people() == 2
    return "the same face id: the same person, no duplicate"


@case
def test_empty_update_never_wipes_a_fact():
    m = fresh()
    p, _ = m.person("face-v")
    m.update_notes(p, "They are building a robot arm.", KV)
    m.update_notes(p, "", KV)
    m.update_notes(p, "   ", KV)
    notes = m.notes(p, KV)
    assert notes == "They are building a robot arm.", notes
    return "empty updates ignored, the fact stays"


@case
def test_notes_are_only_for_their_key():
    path = Path(tempfile.mkdtemp()) / "mem.sqlite"
    m = MemoryStore(path)
    p, _ = m.person("face-v")
    m.update_notes(p, "They are building a robot arm.", KV)
    m.add_episode(p, 0, 10, "Alessandro asked about the camera.", KV)
    raw = path.read_bytes()
    assert b"robot arm" not in raw and b"camera" not in raw and b"Alessandro" not in raw
    try:
        m.notes(p, KM)
    except InvalidTag:
        return "nothing readable in the file, nothing readable with another key"
    raise AssertionError("notes opened with another person's key")


@case
def test_no_memory_crosses_between_people():
    m = fresh()
    v, _ = m.person("face-v")
    s, _ = m.person("face-m")
    m.update_notes(v, "They are building a robot arm.", KV)
    m.update_notes(s, "They are preparing a thesis defence in March.", KM)
    m.add_episode(v, 0, 10, "Alessandro asked about the camera.", KV)
    m.add_episode(s, 0, 10, "Maria said she is nervous about the defence.", KM)
    cv, cs = m.context_for(v, "Alessandro", KV), m.context_for(s, "Maria", KM)
    for leak in ("Maria", "thesis", "defence", "nervous"):
        assert leak not in cv, f"Alessandro's context contains '{leak}':\n{cv}"
    for leak in ("Alessandro", "robot arm", "camera"):
        assert leak not in cs, f"Maria's context contains '{leak}':\n{cs}"
    return "each person's context holds only their own facts"


@case
def test_context_stays_within_budget():
    m = fresh()
    p, _ = m.person("face-v")
    m.update_notes(p, " ".join(["They are building a robot arm and care a lot "
                                "about servo calibration."] * 40), KV)
    for budget in (40, 120, 300):
        ctx = m.context_for(p, "Alessandro", KV, budget_tokens=budget)
        used = sum(approx_tokens(l) for l in ctx.split("\n"))
        assert used <= budget + 2, f"budget {budget}, used {used}"
    return "very long notes: context always within 40 / 120 / 300 tokens"


@case
def test_context_has_when_but_no_narratives():
    m = fresh()
    p, _ = m.person("face-v")
    m.update_notes(p, "Alessandro is building a robot arm.", KV)
    now = time.time()
    m.add_episode(p, now - 3 * 86400, now - 3 * 86400, "OLD: Alessandro said he likes gears.", KV)
    m.add_episode(p, now - 86400, now - 86400, "RECENT: Alessandro said he is building an arm.", KV)
    ctx = m.context_for(p, "Alessandro", KV, now=now)
    assert ctx.startswith("You have met Alessandro before") and "most recently yesterday" in ctx, ctx
    assert "OLD" not in ctx and "RECENT" not in ctx and "said" not in ctx, ctx
    assert "robot arm" in ctx, ctx
    return "profile + 'most recently yesterday'; no episode narratives"


@case
def test_empty_encounter_is_not_stored():
    m = fresh()
    p, _ = m.person("face-v")
    assert m.add_episode(p, 0, 1, "  ", KV) is None
    assert not m.recent_episodes(p, KV)
    return "episode with no content not saved"


@case
def test_memory_survives_a_restart():
    path = Path(tempfile.mkdtemp()) / "mem.sqlite"
    m = MemoryStore(path)
    p, _ = m.person("face-v")
    m.update_notes(p, "They are building a robot arm.", KV)
    m.add_episode(p, 0, 1, "First meeting.", KV)
    m.close()
    m2 = MemoryStore(path)
    q, created = m2.person("face-v")
    assert not created and m2.notes(q, KV) == "They are building a robot arm."
    assert [e["summary"] for e in m2.recent_episodes(q, KV)] == ["First meeting."]
    return "profile and episodes found again after reopening"


@case
def test_unknown_person_has_no_context():
    m = fresh()
    assert m.context_for(12345, "Nobody", KV) == "" and not m.forget("face-x")
    return "person never seen: no context"


@case
def test_forgotten_means_gone():
    m = fresh()
    v, _ = m.person("face-v")
    s, _ = m.person("face-m")
    m.update_notes(v, "They are building a robot arm.", KV)
    m.add_episode(v, 0, 1, "First meeting.", KV)
    m.add_episode(s, 0, 1, "Maria came by.", KM)
    assert m.forget("face-v")
    n_eps = m.db.execute("SELECT COUNT(*) FROM episodes WHERE person_id = ?", (v,)).fetchone()[0]
    assert m.people() == 1 and n_eps == 0 and len(m.recent_episodes(s, KM)) == 1
    return "forget: their profile and episodes gone, the others' untouched"


@case
def test_a_new_name_is_written_into_notes_and_episodes():
    m = fresh()
    v, _ = m.person("face-v")
    m.update_notes(v, "Alessandro is building a robot arm. Alessandro's arm has six joints.", KV)
    m.add_episode(v, 0, 1, "ALESSANDRO introduced himself.", KV)
    m.rename(v, "Alessandro", "Luca", KV)
    notes, ep = m.notes(v, KV), m.recent_episodes(v, KV)[0]["summary"]
    assert notes == "Luca is building a robot arm. Luca's arm has six joints." and ep == "Luca introduced himself.", (notes, ep)
    return f"renamed: {notes!r}"


@case
def test_relative_time_wording():
    got = [when(s) for s in (120, 2 * 3600, 86400 + 60, 4 * 86400, 20 * 86400, 70 * 86400)]
    expected = ["a few minutes ago", "earlier today", "yesterday", "4 days ago",
                "2 weeks ago", "2 months ago"]
    assert got == expected, got
    return ", ".join(got)


@case
def test_self_facts_add_list_remove():
    m = fresh()
    a = m.add_self_fact("Your hand was replaced.", now=100.0)
    b = m.add_self_fact("  There is an open day   on Friday. ", now=200.0)
    texts = [f["text"] for f in m.self_facts()]
    assert texts == ["There is an open day on Friday.", "Your hand was replaced."], texts
    assert m.remove_self_fact(a) and not m.remove_self_fact(a)
    assert [f["id"] for f in m.self_facts()] == [b]
    try:
        m.add_self_fact("   ")
    except ValueError:
        return "add, list (newest first), remove; empty refused"
    raise AssertionError("an empty fact was accepted")


@case
def test_self_context_names_nobody():
    m = fresh()
    now = start_of_day(time.time()) + 15 * 3600
    for name, key in (("Alessandro", KV), ("Maria", KM)):
        p, _ = m.person(f"face-{name}")
        m.update_notes(p, f"{name} is building a robot arm.", key)
        m.add_episode(p, now - 3600, now - 3000, f"{name} came by.", key)
    ctx = m.self_context(now=now)
    assert "2 people" in ctx, ctx
    assert "Alessandro" not in ctx and "Maria" not in ctx and "arm" not in ctx, ctx
    return "counts yes, people's names and facts no"


@case
def test_self_context_counts_only_today():
    m = fresh()
    now = start_of_day(time.time()) + 15 * 3600
    v, _ = m.person("face-v")
    m.add_episode(v, now - 86400, now - 86000, "yesterday", KV)
    m.add_episode(v, now - 600, now - 500, "today", KV)
    m.add_episode(v, now - 300, now - 200, "today again", KV)
    ctx = m.self_context(now=now).splitlines()[0]
    assert ctx.startswith("Before this conversation, 1 person has"), ctx
    return ctx


@case
def test_self_context_stays_within_budget():
    m = fresh()
    for i in range(40):
        m.add_self_fact(f"Fact number {i} about your hardware and your software.", now=float(i))
    ctx = m.self_context(now=1000.0, budget_tokens=150)
    assert approx_tokens(ctx) <= 150, approx_tokens(ctx)
    assert "Fact number 39" in ctx and "Fact number 0 " not in ctx, ctx
    return f"40 facts, {approx_tokens(ctx)} tokens in the context, the newest ones"


def main():
    width = max(len(c.__name__) for c in CASES)
    failed = 0
    for c in CASES:
        try:
            print(f"  PASS  {c.__name__:<{width}}  {c()}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {c.__name__:<{width}}  {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
