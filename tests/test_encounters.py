#!/usr/bin/env python3
"""Tests for encounters and the Mind orchestration.

No models and no robot: the model calls are replaced by stand-ins that behave
like two_stage in the one respect that matters here - a board reading
"MY NAME IS X" puts X into the working record, AFTER the reply is decided.
Faces are labels ("face-V"), and FakeFaces stands in for pi/face_key: two
frames of a label that was enrolled open it.
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
from encounters import CONSENT_QUESTION, SAY, UPDATE_NAME, EncounterTracker, Mind  # noqa: E402
from memory_store import MemoryStore  # noqa: E402

CASES = []
V, M = "face-V", "face-M"


def case(fn):
    CASES.append(fn)
    return fn


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeModels:
    """Stands in for two_stage.  The 'image' is simply the board text."""

    def __init__(self):
        self.calls = []            # (board, memory) for every decide()
        self.records = []          # the record as decide() received it

    def decide(self, image, record, memory, is_new=None, intercept=None):
        self.calls.append((image, memory))
        self.records.append(dict(record))
        seen = {"writing": image}
        if is_new is not None and not is_new(seen):
            return {"action": "none", "repeat": True}, seen
        if intercept is not None and image and (taken := intercept(seen)) is not None:
            return taken, seen
        act = {"action": "speak", "text": f"reply (remembers={bool(memory)})"}
        if image.startswith("MY NAME IS "):
            record["person_name"] = image[len("MY NAME IS "):].title()
        if image.startswith("I AM BUILDING "):
            record["person_notes"] = "They are building " + image[len("I AM BUILDING "):].lower() + "."
        return act, seen

    def summarise(self, name, old_notes, turns):
        boards = [w for w, _ in turns if w]
        notes = " ".join(x for x in [old_notes] + [
            "They are building " + b[len("I AM BUILDING "):].lower() + "."
            for b in boards if b.startswith("I AM BUILDING ")] if x)
        return f"{name} wrote {len(boards)} boards.", notes

    def reply_fresh(self, seen, record):
        return f"fresh reply to {record['person_name']}"

    def reply_with_memory(self, seen, record, memory):
        return f"welcome back ({memory.splitlines()[0] if memory else 'first time'})"


class FakeFaces:
    def __init__(self):
        self.people = {}           # label -> person

    def __len__(self):             # as FaceKeys: nobody remembered yet is falsy
        return len(self.people)

    def recognise(self, frames):
        if len(frames) < 2 or len(set(frames)) != 1 or frames[0] not in self.people:
            return None
        return SimpleNamespace(**{**vars(self.people[frames[0]]), "known": True})

    def enrol(self, frames, name):
        found = self.recognise(frames)
        if found:
            return found
        self.people[frames[-1]] = SimpleNamespace(id=f"id-{frames[-1]}", name=name, errors=3,
                                                  memory_key=os.urandom(32), known=False)
        return self.people[frames[-1]]

    def rename(self, face_id, name):
        for p in self.people.values():
            if p.id == face_id:
                p.name = name

    def forget(self, face_id):
        self.people = {k: p for k, p in self.people.items() if p.id != face_id}

    def knows_name(self, name):
        name = name.casefold()
        return any(name in {p.name.casefold(), *p.name.casefold().split()}
                   for p in self.people.values())


def setup(timeout_s=5.0, same_person=None, confirm=False, careful=None, faces=True, debug=False):
    clock, models = Clock(), FakeModels()
    store = MemoryStore(Path(tempfile.mkdtemp()) / "mem.sqlite")
    mind = Mind(store, EncounterTracker(timeout_s=timeout_s, same_person=same_person),
                models.decide, models.summarise, models.reply_with_memory, clock=clock,
                reply_fresh=models.reply_fresh,
                confirm_names=(lambda w, n: w.upper().endswith(n.upper())) if confirm else None,
                summarise_careful=careful, background=careful is not None,
                faces=FakeFaces() if faces else None, debug=debug)
    return mind, store, models, clock


def say(mind, clock, board, dt=1.0, face=True, vec=None):
    clock.advance(dt)
    return mind.on_frame(face, board, vec)


def silence(mind, clock, seconds):
    for _ in range(int(seconds)):
        clock.advance(1.0)
        mind.on_frame(False, "")


def arrive(mind, clock, vec):
    """A face comes into view: John takes one frame to look at it."""
    act = say(mind, clock, "", vec=vec)
    assert act.get("looking"), act
    return act


def meet(mind, clock, name, vec, *boards):
    """They arrive, introduce themselves, say yes, write `boards`."""
    arrive(mind, clock, vec)
    say(mind, clock, f"MY NAME IS {name.upper()}", vec=vec)
    act = say(mind, clock, "YES", vec=vec)
    assert act.get("text") == SAY["remembered"].format(name=name), act
    for b in boards:
        say(mind, clock, b, vec=vec)


def stored(mind, store, vec):
    """(notes, episodes) of the person with this face, opened with their key."""
    p = mind.faces.people[vec]
    pid, created = store.person(p.id)
    assert not created, "not in the memory store"
    return store.notes(pid, p.memory_key), store.recent_episodes(pid, p.memory_key, 10)


def episodes(store):
    return store.db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]


@case
def test_no_face_no_encounter():
    mind, store, models, clock = setup()
    silence(mind, clock, 10)
    assert mind.tracker.active is None and not models.calls
    return "10s with no faces: no encounter, no model calls"


@case
def test_short_glance_away_keeps_the_encounter():
    mind, store, models, clock = setup(timeout_s=5)
    say(mind, clock, "HELLO")
    silence(mind, clock, 3)
    say(mind, clock, "STILL ME")
    enc = mind.tracker.active
    assert enc is not None and len(enc.turns) == 2, enc
    return "3s away, under the 5s timeout: same encounter, 2 turns"


@case
def test_anonymous_encounter_leaves_no_trace():
    mind, store, models, clock = setup(timeout_s=5)
    say(mind, clock, "HELLO", vec=V)
    say(mind, clock, "WHAT DO YOU SEE?", vec=V)
    silence(mind, clock, 8)
    assert store.people() == 0 and episodes(store) == 0 and not mind.faces.people
    assert mind.log[-1][0] == "discarded", mind.log
    return "encounter with no name: nothing archived"


@case
def test_an_introduction_is_asked_for_consent():
    mind, store, models, clock = setup(timeout_s=5)
    arrive(mind, clock, V)
    act = say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    assert act["text"] == f"fresh reply to Alessandro {CONSENT_QUESTION}", act
    assert not mind.faces.people and store.people() == 0
    return f"greeted, then: {CONSENT_QUESTION!r}; nothing kept yet"


@case
def test_a_yes_remembers_the_face_and_the_encounter():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    notes, eps = stored(mind, store, V)
    assert "robot arm" in notes and len(eps) == 1, (notes, eps)
    wrote = eps[0]["summary"]
    assert wrote == "Alessandro wrote 2 boards.", wrote
    return f"face enrolled, notes {notes!r}, 1 episode (the YES not in it)"


@case
def test_a_no_leaves_nothing():
    mind, store, models, clock = setup(timeout_s=5)
    arrive(mind, clock, V)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    act = say(mind, clock, "NO THANKS", vec=V)
    say(mind, clock, "I AM BUILDING A ROBOT ARM", vec=V)
    silence(mind, clock, 8)
    assert act["text"] == SAY["declined"], act
    assert store.people() == 0 and episodes(store) == 0 and not mind.faces.people
    assert mind.log[-1][:2] == ("discarded", "not remembered: Alessandro"), mind.log
    return "declined: talked to, nothing kept"


@case
def test_consent_is_the_next_board_or_nothing():
    mind, store, models, clock = setup(timeout_s=5)
    arrive(mind, clock, V)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    act = say(mind, clock, "WHAT CAN YOU DO?", vec=V)
    later = say(mind, clock, "YES", vec=V)
    assert act["text"].startswith("reply") and later["text"].startswith("reply"), (act, later)
    silence(mind, clock, 8)
    assert store.people() == 0 and not mind.faces.people
    return "question unanswered, a YES later is not a consent: nothing kept"


@case
def test_yes_with_no_face_seen_asks_to_look():
    mind, store, models, clock = setup(timeout_s=60)
    say(mind, clock, "MY NAME IS ALESSANDRO")
    act = say(mind, clock, "YES")                     # face not frontal: no vectors
    assert act["text"] == SAY["look"], act
    say(mind, clock, "YES", vec=V)                     # same board: not answered again
    clock.advance(11)
    act = say(mind, clock, "YES", vec=V)               # written again, looking
    assert act["text"] == SAY["remembered"].format(name="Alessandro"), act
    return "no face to remember yet: asked to look, then remembered"


@case
def test_returning_face_is_recognised_and_greeted():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    clock.advance(2 * 86400)                       # two days later
    models.calls.clear()
    arrive(mind, clock, V)
    assert not models.calls, "answered before the face could be recognised"
    act = say(mind, clock, "", vec=V)
    assert act["text"].startswith("welcome back (You have met Alessandro") and "reply" not in act["text"], act
    assert mind.tracker.active.person_name == "Alessandro"
    say(mind, clock, "WHAT AM I BUILDING?", vec=V)
    memory_on_question = models.calls[-1][1]
    assert "robot arm" in memory_on_question and "days ago" in memory_on_question, memory_on_question
    return "two days later, nothing written: greeted by name, and 'robot arm' reaches the model"


@case
def test_a_dark_face_is_tried_again_with_the_light_evened_out():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V)
    silence(mind, clock, 8)
    clock.advance(86400)
    dark = ("face-V in the dark", V)                 # as seen opens nothing; evened out does
    say(mind, clock, "", vec=dark)
    act = say(mind, clock, "", vec=dark)
    assert act["text"].startswith("welcome back (You have met Alessandro"), act
    return "as seen: nobody; light evened out: Alessandro"


@case
def test_enrolment_uses_the_frames_as_seen():
    mind, store, models, clock = setup(timeout_s=5)
    both = (V, "face-V evened out")
    arrive(mind, clock, both)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=both)
    act = say(mind, clock, "YES", vec=both)
    assert act["text"] == SAY["remembered"].format(name="Alessandro") and V in mind.faces.people, act
    mind2, *_ = setup(timeout_s=5)
    only_evened = (None, "face-M evened out")
    arrive(mind2, clock, only_evened)
    say(mind2, clock, "MY NAME IS MARIA", vec=only_evened)
    act = say(mind2, clock, "YES", vec=only_evened)
    assert act["text"] == SAY["remembered"].format(name="Maria"), act
    assert "face-M evened out" in mind2.faces.people
    return "enrolled from the frames as seen; evened out only when those have no face"


@case
def test_next_person_gets_nothing_of_the_previous():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    models.calls.clear()
    say(mind, clock, "HELLO", vec=M)
    say(mind, clock, "WHAT IS ALESSANDRO BUILDING?", vec=M)
    say(mind, clock, "MY NAME IS MARIA", vec=M)
    say(mind, clock, "WHAT DO YOU KNOW ABOUT ME?", vec=M)
    leaks = [(b, m) for b, m in models.calls if "Alessandro" in m or "robot arm" in m]
    assert not leaks, f"Alessandro's memory passed on during Maria's encounter: {leaks}"
    return "4 turns from Maria, even asking about Alessandro: none of his memory reaches the model"


@case
def test_without_faces_nobody_is_remembered():
    mind, store, models, clock = setup(timeout_s=5, faces=False)
    act = say(mind, clock, "MY NAME IS ALESSANDRO")
    say(mind, clock, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    say(mind, clock, "WHAT IS MY NAME?")
    assert act["text"] == "fresh reply to Alessandro", act
    assert store.people() == 0 and not mind.tracker.active.person_name
    return "no face keys: greeted, not asked, not kept"


@case
def test_face_check_forgives_a_long_gap_for_the_same_face():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    arrive(mind, clock, V)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    silence(mind, clock, 20)                       # well past timeout_s
    say(mind, clock, "WHAT IS MY NAME?", vec=V)
    enc = mind.tracker.active
    assert enc is not None and enc.person_name == "Alessandro", enc
    return "20s away but the same face: same encounter, Alessandro stays Alessandro"


@case
def test_face_check_catches_a_swap_at_once():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    meet(mind, clock, "Alessandro", V)
    say(mind, clock, "WHAT IS MY NAME?", vec=M, dt=0.5)   # swap, no gap
    enc = mind.tracker.active
    assert not enc.person_name and enc.person_id is None, "Maria was taken for Alessandro"
    assert mind.log and mind.log[-1][0] == "queued", mind.log
    silence(mind, clock, 40)
    assert episodes(store) == 1 and mind.log[-1][0] == "archived", mind.log
    return "different face with no pause: encounter closed, archived when nobody is there"


@case
def test_shutdown_does_not_lose_the_last_encounter():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V)
    mind.shutdown()
    assert len(stored(mind, store, V)[1]) == 1
    return "shutdown with an encounter open: archived anyway"


@case
def test_notes_drop_questions_and_other_people():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V)
    silence(mind, clock, 8)
    models.summarise = lambda name, old, turns: (
        "Maria came by.",
        "Maria is writing her thesis. She asked about Alessandro's work. "
        "What is Alessandro building? She is interested in Alessandro's project. "
        "She works in the vision group.")
    mind.summarise = models.summarise
    meet(mind, clock, "Maria", M)
    silence(mind, clock, 8)
    notes = stored(mind, store, M)[0]
    assert "Alessandro" not in notes and "asked" not in notes and "?" not in notes, notes
    assert "thesis" in notes and "vision group" in notes, notes
    return f"notes saved: {notes!r}"


@case
def test_working_record_is_cleaned_during_the_encounter():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V)
    silence(mind, clock, 8)
    real = models.decide

    def noisy(image, record, memory, is_new=None, intercept=None):
        act, seen = real(image, record, memory, is_new, intercept)
        if image == "WHAT IS ALESSANDRO BUILDING?":
            record["person_notes"] = ("She asked what Alessandro is building. "
                                      "She is writing a thesis.")
        return act, seen
    mind.decide = noisy
    arrive(mind, clock, M)
    say(mind, clock, "MY NAME IS MARIA", vec=M)
    say(mind, clock, "WHAT IS ALESSANDRO BUILDING?", vec=M)
    notes = mind.tracker.active.record["person_notes"]
    assert "Alessandro" not in notes and "asked" not in notes, notes
    assert "thesis" in notes, notes
    return f"Maria's working record: {notes!r}"


@case
def test_known_person_record_starts_from_the_archive():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    clock.advance(86400)
    say(mind, clock, "HELLO", vec=V)
    say(mind, clock, "WHAT AM I BUILDING?", vec=V)
    notes = models.records[-1]["person_notes"]
    assert "robot arm" in notes, f"record on the question turn: {notes!r}"
    return f"record when the face is recognised: {notes!r}"


@case
def test_new_name_same_face_starts_a_new_encounter():
    mind, store, models, clock = setup(timeout_s=5, faces=False)
    say(mind, clock, "MY NAME IS ALESSANDRO")
    say(mind, clock, "I AM BUILDING A ROBOT ARM")
    act = say(mind, clock, "MY NAME IS MARIA", dt=0.5)       # board handed over
    assert act["text"] == "fresh reply to Maria", act
    say(mind, clock, "WHAT AM I BUILDING?")
    rec = models.records[-1]
    assert "robot arm" not in rec["person_notes"] and rec["person_name"] == "Maria", rec
    return "Alessandro then Maria with no pause: two encounters, nothing of Alessandro's to Maria"


@case
def test_a_remembered_face_stays_itself_whatever_name_is_written():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    clock.advance(86400)
    say(mind, clock, "HELLO", vec=V)
    say(mind, clock, "HELLO AGAIN", vec=V)
    act = say(mind, clock, "MY NAME IS MARIA", vec=V)
    assert act["text"] == SAY["known as"].format(name="Alessandro"), act
    say(mind, clock, "WHAT AM I BUILDING?", vec=V)
    rec = models.records[-1]
    assert rec["person_name"] == "Alessandro" and "robot arm" in rec["person_notes"], rec
    assert "MY NAME IS MARIA" not in [w for w, _ in mind.tracker.active.turns], "the board kept"
    assert ("name ignored", "Maria: the face is Alessandro's") in mind.log, mind.log
    return "recognised as Alessandro, writes MARIA: still Alessandro, no new person"


def back_next_day(mind, clock, vec):
    """They left; a day later their face is recognised."""
    silence(mind, clock, 8)
    clock.advance(86400)
    arrive(mind, clock, vec)
    act = say(mind, clock, "", vec=vec)
    assert act["text"].startswith("welcome back"), act


@case
def test_the_update_keyword_changes_a_name_after_a_yes():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    back_next_day(mind, clock, V)
    ask = say(mind, clock, "UPDATE MY NAME TO LUCA", vec=V)
    done = say(mind, clock, "YES", vec=V)
    assert ask["text"] == SAY["rename?"].format(name="Luca"), ask
    assert done["text"] == SAY["renamed"].format(name="Luca"), done
    say(mind, clock, "WHAT AM I BUILDING?", vec=V)
    rec = models.records[-1]
    assert rec["person_name"] == "Luca" and "robot arm" in rec["person_notes"], rec
    assert mind.faces.people[V].name == "Luca" and mind.faces.knows_name("Luca") \
        and not mind.faces.knows_name("Alessandro")
    assert not [w for w, _ in mind.tracker.active.turns if "UPDATE" in w or w == "YES"]
    back_next_day(mind, clock, V)
    assert mind.tracker.active.person_name == "Luca"
    return "UPDATE MY NAME TO LUCA, YES: Luca from then on, the notes kept"


@case
def test_an_update_needs_a_yes_as_the_next_board():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V)
    back_next_day(mind, clock, V)
    say(mind, clock, "UPDATE MY NAME TO LUCA", vec=V)
    no = say(mind, clock, "NO", vec=V)
    say(mind, clock, "UPDATE: CALL ME MARCO", vec=V)
    other = say(mind, clock, "WHAT TIME IS IT?", vec=V)
    late = say(mind, clock, "YES", vec=V)
    assert no["text"] == SAY["kept name"].format(name="Alessandro"), no
    assert other["text"].startswith("reply") and late["text"].startswith("reply"), (other, late)
    assert mind.tracker.active.person_name == "Alessandro" and mind.faces.people[V].name == "Alessandro"
    return "NO, or anything but YES next: the name stays"


@case
def test_update_only_for_a_remembered_face_and_only_about_the_name():
    mind, store, models, clock = setup(timeout_s=5)
    arrive(mind, clock, V)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    say(mind, clock, "NO", vec=V)
    stranger = say(mind, clock, "UPDATE MY NAME TO LUCA", vec=V)
    assert stranger["text"] == SAY["no name to update"], stranger
    silence(mind, clock, 8)
    meet(mind, clock, "Maria", M)
    news = say(mind, clock, "UPDATE ME ON THE NEWS", vec=M)
    same = say(mind, clock, "UPDATE MY NAME TO MARIA", vec=M)
    assert news["text"].startswith("reply") and same["text"] == SAY["same name"].format(name="Maria"), (news, same)
    return "not remembered: nothing to update; UPDATE about something else: the models answer it"


@case
def test_the_ways_of_writing_an_update():
    for board, name in (("UPDATE MY NAME TO LUCA", "LUCA"), ("UPDATE MY NAME: LUCA", "LUCA"),
                        ("UPDATE: MY NAME IS LUCA ROSSI", "LUCA ROSSI"), ("update name luca", "luca"),
                        ("UPDATE: CALL ME LUCA.", "LUCA"), ("UPDATE MY NAME AS ANNA MARIA", "ANNA MARIA")):
        got = UPDATE_NAME.search(board)
        assert got and got["name"] == name, (board, got and got["name"])
    for board in ("UPDATE ME ON THE NEWS", "PLEASE UPDATE MY NAME TO LUCA", "MY NAME IS LUCA",
                  "UPDATE THE NAME OF THE FILE TO X", "UPDATE"):
        assert not UPDATE_NAME.search(board), board
    return "six ways it is written; five boards that are not one"


@case
def test_debug_shows_what_is_kept_encrypted():
    mind, store, models, clock = setup(timeout_s=5, debug=True)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    back_next_day(mind, clock, V)
    said = [e[1] for e in mind.log if e[0] == "debug"]
    wanted = ("enrolled Alessandro", "notes were '', now 'They are building a robot arm.'",
              "face as seen opened Alessandro's lock", "3 of 56 face bits", "memory for the prompt")
    assert all(any(w in d for d in said) for w in wanted), said
    quiet, *_ = setup(timeout_s=5)
    meet(quiet, clock, "Alessandro", V)
    assert not [e for e in quiet.log if e[0] == "debug"]
    return f"{len(said)} debug entries; none without debug"


def spoken(acts):
    return sum(a.get("action") == "speak" for a in acts)


@case
def test_forget_my_face_forgets_everything():
    mind, store, models, clock = setup(timeout_s=5)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    clock.advance(86400)
    say(mind, clock, "HELLO", vec=V)
    say(mind, clock, "HELLO AGAIN", vec=V)
    act = say(mind, clock, "PLEASE FORGET MY FACE", vec=V)
    assert act["text"] == SAY["forgotten"], act
    say(mind, clock, "I LIKE GEARS", vec=V)
    silence(mind, clock, 8)
    assert store.people() == 0 and episodes(store) == 0 and not mind.faces.people
    say(mind, clock, "", vec=V)
    say(mind, clock, "", vec=V)
    assert not mind.tracker.active.person_name
    return "forgotten: face, notes, episodes, and this encounter; not recognised again"


@case
def test_forget_me_from_a_stranger():
    mind, store, models, clock = setup(timeout_s=5)
    arrive(mind, clock, V)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    act = say(mind, clock, "FORGET ME", vec=V)
    after = say(mind, clock, "YES", vec=V)
    assert act["text"] == SAY["nothing"] and after["text"].startswith("reply"), (act, after)
    silence(mind, clock, 8)
    assert store.people() == 0
    return "nothing to forget, and the question is off"


@case
def test_forget_takes_the_write_up_not_yet_done():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    say(mind, clock, "HELLO", vec=M, dt=0.5)                 # Alessandro's encounter queued
    say(mind, clock, "HELLO", vec=V, dt=0.5)
    say(mind, clock, "HI", vec=V)
    say(mind, clock, "FORGET MY FACE", vec=V)
    silence(mind, clock, 40)
    assert store.people() == 0 and episodes(store) == 0 and not mind.pending
    return "back and forgotten before the write-up: nothing written"


@case
def test_a_board_held_up_is_answered_once():
    mind, store, models, clock = setup(timeout_s=5)
    acts = [say(mind, clock, "WHAT IS YOUR NAME?") for _ in range(4)]
    acts.append(say(mind, clock, "WHAT IS YOUR NAME"))   # read without the "?"
    assert spoken(acts) == 1, acts
    assert len(mind.tracker.active.turns) == 1, mind.tracker.active.turns
    return "5 frames of the same board (one read without '?'): 1 reply, 1 turn"


@case
def test_my_and_your_are_different_boards():
    mind, store, models, clock = setup(timeout_s=5)
    acts = [say(mind, clock, "WHAT IS MY NAME?"), say(mind, clock, "WHAT IS YOUR NAME?")]
    assert spoken(acts) == 2, acts
    return "WHAT IS MY NAME? then WHAT IS YOUR NAME?: two replies"


@case
def test_a_new_board_is_answered():
    mind, store, models, clock = setup(timeout_s=5)
    acts = [say(mind, clock, b) for b in
            ("MY NAME IS ALESSANDRO", "MY NAME IS ALESSANDRO", "NO",
             "I AM BUILDING A ROBOT ARM", "I AM BUILDING A ROBOT ARM", "WHAT AM I BUILDING?")]
    assert spoken(acts) == 4, acts
    return "4 different boards, each held for 1-2 frames: 4 replies"


@case
def test_a_board_shown_again_later_is_answered_again():
    mind, store, models, clock = setup(timeout_s=60)
    first = say(mind, clock, "WHAT IS YOUR NAME?")
    for _ in range(15):                       # board lowered, still there, 15 s
        say(mind, clock, "")
    again = say(mind, clock, "WHAT IS YOUR NAME?")
    assert spoken([first, again]) == 2, (first, again)
    return "the same board shown again after 15 s: answered again"


@case
def test_someone_just_looking_is_spoken_to_once_then_after_a_while():
    mind, store, models, clock = setup(timeout_s=60)
    acts = [say(mind, clock, "") for _ in range(29)]
    assert spoken(acts) == 1, spoken(acts)
    acts = [say(mind, clock, "") for _ in range(2)]
    assert spoken(acts) == 1, spoken(acts)
    return "30 s without writing: one sentence at the start, the second after 30 s"


@case
def test_switch_board_is_not_answered_twice():
    mind, store, models, clock = setup(timeout_s=5, faces=False)
    say(mind, clock, "MY NAME IS ALESSANDRO")
    acts = [say(mind, clock, "MY NAME IS MARIA", dt=0.5) for _ in range(3)]
    assert spoken(acts) == 1, acts
    return "the board that switches person: a single reply, even after the switch"


@case
def test_archiving_waits_until_nobody_is_there():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    summaries = []
    real = mind.summarise
    mind.summarise = lambda *a: (summaries.append(clock()), real(*a))[1]
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    say(mind, clock, "HELLO", vec=M, dt=0.5)          # Maria steps in
    say(mind, clock, "WHAT DO YOU SEE?", vec=M)
    assert not summaries, "write-up done while Maria was talking"
    silence(mind, clock, 40)
    assert len(summaries) == 1 and stored(mind, store, V)[0], summaries
    return "Alessandro's write-up done only after Maria left"


@case
def test_coming_back_before_the_write_up_still_gets_it():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    say(mind, clock, "HELLO", vec=M, dt=0.5)          # someone else, briefly
    say(mind, clock, "HELLO", vec=V, dt=0.5)          # Alessandro again
    act = say(mind, clock, "HI", vec=V)               # recognised, with a board
    assert act["text"] == "reply (remembers=True)", act
    say(mind, clock, "WHAT AM I BUILDING?", vec=V)
    rec = models.records[-1]
    assert "robot arm" in rec["person_notes"], rec
    assert ("archived early", "Alessandro") in [e[:2] for e in mind.log], mind.log
    return "back before the write-up was done: it is done at once and used"


@case
def test_archive_drops_states_and_dates_relative_time():
    mind, store, models, clock = setup(timeout_s=5)
    seen_turns = []
    mind.summarise = lambda name, old, turns: (seen_turns.extend(turns), (
        "Maria came by.",
        "Maria is tired today. Maria is leaving the lab next month. Maria likes optics."))[1]
    meet(mind, clock, "Maria", M, "I AM TIRED TODAY, I LIKE OPTICS")
    silence(mind, clock, 8)
    notes = stored(mind, store, M)[0]
    assert "tired" not in notes and "optics" in notes, notes
    assert "next month (as of " in notes, notes
    wrote = " ".join(w for w, _ in seen_turns)
    assert "TIRED" not in wrote and "OPTICS" in wrote, wrote
    return f"archive: {notes!r}"


@case
def test_after_a_long_pause_nothing_is_said_until_the_face_is_checked():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    arrive(mind, clock, V)
    say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    silence(mind, clock, 20)                       # rubbing out, writing
    n = len(models.calls)
    act = say(mind, clock, "", vec=None)           # back, head still down
    assert act.get("hold") and len(models.calls) == n, (act, len(models.calls) - n)
    act = say(mind, clock, "WHAT IS MY NAME?", vec=V)
    enc = mind.tracker.active
    assert act["action"] == "speak" and enc.person_name == "Alessandro", (act, enc)
    return "after 20 s, head down: silence; same frontal face: same encounter"


@case
def test_after_a_long_pause_another_face_is_a_new_encounter():
    same = lambda a, b: a == b
    mind, store, models, clock = setup(timeout_s=5, same_person=same)
    meet(mind, clock, "Alessandro", V)
    silence(mind, clock, 20)
    say(mind, clock, "", vec=None)
    say(mind, clock, "WHAT IS MY NAME?", vec=M)
    enc = mind.tracker.active
    assert not enc.person_name and not models.calls[-1][1], (enc, models.calls[-1])
    return "after 20 s a different face appears: new encounter, no memory"


def introduced(mind):
    return [e[1] for e in mind.log if e[0] == "introduced"]


@case
def test_a_name_read_once_is_not_taken():
    mind, store, models, clock = setup(timeout_s=5, confirm=True, faces=False)
    say(mind, clock, "MY NAME IS ALESSANDRO")
    say(mind, clock, "MY NAME IS ALESSANDRO")          # confirmed: Alessandro
    first = say(mind, clock, "MY NAME IS BRIAN")     # misread, one frame
    say(mind, clock, "")
    say(mind, clock, "WHAT IS MY NAME?")
    enc = mind.tracker.active
    assert first["action"] == "none" and introduced(mind) == ["Alessandro"], (first, mind.log)
    assert enc.person_name == "Alessandro" and "BRIAN" not in " ".join(w for w, _ in enc.turns), enc
    return "name read on a single frame: not taken, Alessandro's encounter intact"


@case
def test_a_name_read_twice_is_taken_and_greeted():
    mind, store, models, clock = setup(timeout_s=5, confirm=True)
    arrive(mind, clock, V)
    a1 = say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    a2 = say(mind, clock, "MY NAME IS ALESSANDRO", vec=V)
    assert a1["action"] == "none" and a2.get("text") == f"fresh reply to Alessandro {CONSENT_QUESTION}", (a1, a2)
    assert introduced(mind) == ["Alessandro"]
    return "same name on two frames: greeted, and asked, on the second"


@case
def test_a_misread_name_gives_way_to_the_right_one():
    mind, store, models, clock = setup(timeout_s=5, confirm=True, faces=False)
    say(mind, clock, "MY NAME IS ALEXA")
    say(mind, clock, "MY NAME IS ALEXANDER")
    say(mind, clock, "MY NAME IS ALEXANDER")
    assert introduced(mind) == ["Alexander"], mind.log
    return "misread once, read right twice: only the right name"


def slow_careful(seconds, log):
    """A careful write-up that takes `seconds` and gives up when told to."""
    def run(name, old, turns, stop):
        log.append(name)
        t0 = time.time()
        while time.time() - t0 < seconds:
            if stop.is_set():
                return None
            time.sleep(0.005)
        return f"{name} came by (careful).", f"Careful notes about {name}."
    return run


@case
def test_careful_writeup_stops_when_someone_arrives_and_resumes_later():
    runs = []
    mind, store, models, clock = setup(timeout_s=5, careful=slow_careful(0.4, runs))
    meet(mind, clock, "Alessandro", V)
    silence(mind, clock, 8)                        # gone: write-up starts
    time.sleep(0.05)
    t0 = time.time()
    say(mind, clock, "HELLO", vec=M)               # someone else, right away
    mind._worker.join(2)
    stopped_in = time.time() - t0
    assert not stored(mind, store, V)[1] and len(mind.pending) == 1, "written while someone was there"
    assert stopped_in < 0.3, stopped_in
    silence(mind, clock, 8)                        # nobody again: it resumes
    mind._worker.join(2)
    notes = stored(mind, store, V)[0]
    assert notes == "Careful notes about Alessandro." and not mind.pending, (notes, mind.pending)
    return f"stopped in {stopped_in * 1000:.0f} ms, resumed and finished: {notes!r}"


@case
def test_back_during_the_careful_writeup_gets_the_fast_one_at_once():
    runs = []
    mind, store, models, clock = setup(timeout_s=5, careful=slow_careful(5.0, runs))
    meet(mind, clock, "Alessandro", V, "I AM BUILDING A ROBOT ARM")
    silence(mind, clock, 8)
    time.sleep(0.05)
    t0 = time.time()
    say(mind, clock, "HELLO", vec=V)               # back while it is being written
    say(mind, clock, "HELLO", vec=V)               # recognised
    took = time.time() - t0
    notes = stored(mind, store, V)[0]
    assert took < 0.5 and "robot arm" in notes and "Careful" not in notes, (took, notes)
    return f"back during the careful write-up: fast one in {took * 1000:.0f} ms, notes {notes!r}"


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
