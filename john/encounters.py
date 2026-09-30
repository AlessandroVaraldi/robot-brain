#!/usr/bin/env python3
"""Encounters: when a conversation starts, who it is with, and what is kept.

Policy:

* A person is a face, and only with their consent.  Someone who introduces
  themselves is asked "May I remember your face?"; on a yes their face becomes
  a lock that only it opens (pi/face_key), and from then on they are
  recognised by it.  Their name is something the robot knows about that face,
  not who they are: a recognised face stays that person whatever name is
  written.  "FORGET MY FACE" forgets them.
* Without a yes, nothing outlives the encounter: no face, no name, no notes.
* An encounter ends when the face has been gone for a while, or, with face
  comparison on (--face-check), when the face that comes back is a different
  one.  Face vectors live only in RAM, only for the length of one encounter.
* A remembered person's encounter becomes an episode, and their notes are
  updated, encrypted under the key their face gives.  That write happens after
  the encounter, when nobody is waiting for a reply.
* Memory about a person reaches the prompt only while that person is the one in
  the encounter.  Nothing about anyone else is ever read.

The model calls are passed in (`decide`, `summarise`), so everything in this file
can be tested with stand-ins and no GPU.
"""

from __future__ import annotations

import difflib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from memory_store import name_key

CONSENT_QUESTION = "May I remember your face?"
YES = re.compile(r"^\W*(yes|yeah|yep|sure|ok|okay|of course|please do|go ahead|y)\b", re.I)
NO = re.compile(r"^\W*(no|nope|nah|don'?t|please don'?t|n)\b", re.I)
FORGET = re.compile(r"\bforget\s+(me|my\s+face|about\s+me|who\s+i\s+am)\b", re.I)
# A remembered person changes their name: the keyword, then the name - the
# whole board, so that "UPDATE ME ON THE NEWS" is not a new name.
UPDATE_NAME = re.compile(
    r"^\W*update\b[\s:,\-]*(?:my\s+name\s+is|call\s+me|(?:my\s+)?name(?:\s+(?:to|is|as))?)"
    r"[\s:,\-]+(?P<name>[a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,2})[\s.!]*$", re.I)
SAY = {
    "remembered": "Thank you, {name}. I will remember your face.",
    "already": "I already know your face, {name}.",
    "look": "I need to see your face clearly: look at me and write YES again.",
    "declined": "All right: I will not remember you after you go.",
    "forgotten": "Done: I have forgotten your face and everything you told me.",
    "nothing": "I do not remember your face, so there is nothing to forget.",
    "failed": "Something went wrong and I could not remember your face.",
    "known as": "I know your face as {name}. If that is not your name, write UPDATE MY NAME TO "
                "and your name.",
    "rename?": "Should I call you {name} from now on?",
    "renamed": "All right, {name}: that is your name from now on.",
    "kept name": "All right, I will keep calling you {name}.",
    "same name": "That is already your name, {name}.",
    "no name to update": "I do not remember your face, so there is no name to update.",
}

# A state of the moment, not a fact about the person: kept in the working
# record for the encounter, never archived, or "I AM TIRED TODAY" becomes
# "Maria is tired" for good.
TRANSIENT =re.compile(r"\b(today|tonight|this (morning|afternoon|evening)|right now|"
                       r"at the moment)\b", re.IGNORECASE)
# Time said relative to the day it was said: archived with that day, or
# "leaving next month" stays next month forever.
RELATIVE = re.compile(r"\b(tomorrow|yesterday|next (week|month|year)|last (week|month|year)|"
                      r"soon|in (a|two|three|\d+) (days?|weeks?|months?))\b", re.IGNORECASE)


def without(writing, pattern):
    """The writing minus the pieces (between punctuation) that match."""
    parts = re.findall(r"[^.!?,;:\n]+[.!?,;:]?", writing or "")
    return " ".join(p.strip() for p in parts if p.strip() and not pattern.search(p))


def for_archive(notes, now):
    """Drop states of the moment; date what is said relative to today."""
    day = time.strftime("%d %b %Y", time.localtime(now))
    kept = []
    for s in re.split(r"(?<=[.!?])\s+", (notes or "").strip()):
        if not s or TRANSIENT.search(s):
            continue
        if RELATIVE.search(s) and "(as of " not in s:
            s = s.rstrip(".") + f" (as of {day})."
        kept.append(s)
    return " ".join(kept)


@dataclass
class Encounter:
    started_at: float
    last_face_at: float
    face_ref: Optional[object] = None      # an embedding, RAM only, this encounter only
    record: dict = field(default_factory=lambda: {"person_name": "", "person_notes": "",
                                                  "pending": ""})
    person_id: Optional[int] = None        # in the memory store: only for a remembered face
    person_name: str = ""
    face_id: str = ""                      # the id their face opened (face_key)
    memory_key: bytes = b""                # the key to their memory; RAM only
    frames: list = field(default_factory=list)   # frontal face vectors, RAM only
    consent: str = ""                      # "", "asked", "yes", "no"
    to_greet: bool = False                 # recognised: greet them by name
    name_ignored: bool = False             # another name written in front of a known face
    rename_to: str = ""                    # a new name asked for, to be confirmed
    memory: str = ""                       # long-term context about THIS person
    turns: list = field(default_factory=list)   # (what they wrote, what the robot said)
    needs_check: bool = False              # back after a long pause, face not yet compared
    pending_name: str = ""                 # a new name read once, not yet confirmed
    pending_at: float = 0.0
    pending_record: dict = field(default_factory=dict)
    last_writing: str = ""                 # last board acted on, normalised
    last_writing_at: float = 0.0           # when it was last seen
    last_spoke_at: Optional[float] = None  # when the robot last said or did something


class EncounterTracker:
    """Decides when an encounter starts and ends.  Pure logic, no models.

    Without a face comparison, the only signal is time: a gap longer than
    `timeout_s` ends the encounter.  That has to stay short, because two people
    swapping inside the window would be taken for one - and the robot might
    repeat what the first told it.

    With a comparison (`same_person`), a face that comes back is checked against
    the one that started the encounter, so a gap can be forgiven for much longer
    (`reappear_s`) when it is the same face, and a different face ends the
    encounter at once, whatever the gap.
    """

    def __init__(self, timeout_s: float = 5.0,
                 same_person: Optional[Callable[[object, object], bool]] = None,
                 reappear_s: float = 30.0):
        self.timeout_s = timeout_s
        self.same_person = same_person
        self.reappear_s = reappear_s
        self.active: Optional[Encounter] = None

    def observe(self, t: float, face: bool, face_vec=None):
        """Feed one frame.  Returns (event, finished_encounter_or_None).

        event is one of: None, "started", "ended", "switched" (ended + started),
        "hold" (someone is back after a long pause but their face could not be
        compared yet: nothing should be said to them until it has been).
        """
        enc = self.active
        if enc is None:
            if face:
                self.active = Encounter(started_at=t, last_face_at=t, face_ref=face_vec)
                return "started", None
            return None, None

        gap = t - enc.last_face_at
        if not face:
            limit = self.reappear_s if self.same_person else self.timeout_s
            if gap > limit:
                self.active = None
                return "ended", enc
            return None, None

        # A face is present.
        if self.same_person and enc.face_ref is not None:
            if face_vec is not None:
                if not self.same_person(enc.face_ref, face_vec):
                    self.active = Encounter(started_at=t, last_face_at=t, face_ref=face_vec)
                    return "switched", enc
                enc.needs_check = False
            elif gap > self.timeout_s:
                # Back after a long pause (rubbing out the board and writing the
                # next message takes a while) but head down, so the face cannot
                # be compared.  Keep the encounter, say nothing, and let the
                # first frontal face decide.
                enc.needs_check = True
        elif gap > self.timeout_s:
            # No comparison available: after a gap, assume it may be someone else.
            self.active = Encounter(started_at=t, last_face_at=t, face_ref=face_vec)
            return "switched", enc
        if enc.face_ref is None and face_vec is not None:
            enc.face_ref = face_vec
        enc.last_face_at = t
        return ("hold" if enc.needs_check else None), None

    def split(self, t: float):
        """End the open encounter now and start another with the same face -
        someone new has introduced themselves without leaving the frame."""
        old = self.active
        self.active = Encounter(started_at=t, last_face_at=t,
                                face_ref=old.face_ref if old else None,
                                frames=list(old.frames) if old else [])
        return old

    def flush(self):
        """End whatever is open - at shutdown, so a last encounter is not lost."""
        enc, self.active = self.active, None
        return enc


class Mind:
    """Ties the eye, the reasoning, the encounters and the archive together.

    decide(image_b64, record, memory, is_new[, intercept]) -> (action, seen)
        one turn of two_stage: look, answer if needed, update the record.
        is_new(seen) says whether what it saw still needs an answer;
        intercept(seen), given only with `faces`, answers the consent question
        and "forget my face" instead of the models.
    summarise(name, old_notes, turns) -> (summary, notes)
        the end-of-encounter write-up
    reply_with_memory(seen, record, memory) -> text
        used once, when a face is recognised, so the greeting can already
        draw on what the robot remembers of them
    faces
        pi/face_key's FaceKeys, or anything with recognise(frames),
        enrol(frames, name), forget(face_id) and knows_name(name).  Without
        it nobody is remembered.
    reply_fresh(seen, record) -> text
        the reply on the turn someone new introduces themselves.  The reply
        decided this turn used the record from before it (empty, or the
        previous person's after a switch), so "MY NAME IS ALESSANDRO" could be
        answered "I don't know who Alessandro is".
    """

    def __init__(self, store, tracker: EncounterTracker, decide, summarise,
                 reply_with_memory=None, clock: Callable[[], float] = time.time,
                 reply_fresh=None, repeat_window_s: float = 10.0,
                 prompt_again_s: float = 30.0, confirm_names=None, continue_window_s: float = 15.0,
                 confirm_window_s: float = 10.0,
                 memory_budget_tokens: int = 300, summarise_careful=None,
                 background: bool = False, faces=None, debug: bool = False):
        self.store = store
        self.faces = faces
        self.debug = debug             # log what is otherwise only kept encrypted
        self.tracker = tracker
        self.decide = decide
        self.summarise = summarise
        self.reply_with_memory = reply_with_memory
        self.reply_fresh = reply_fresh
        self.confirm_names = confirm_names
        self.confirm_window_s = confirm_window_s
        self.repeat_window_s = repeat_window_s
        self.prompt_again_s = prompt_again_s
        self.continue_window_s = continue_window_s   # a board read half, then whole
        self.clock = clock
        self.budget = memory_budget_tokens
        self.log: list = []
        self.last_seen: dict = {}              # what the eye read on the last frame
        self.pending: list = []                # closed, named, not yet archived
        # The careful write-up (summarise_careful(name, old, turns, stop) ->
        # (summary, notes), or None if stopped) runs in the background while
        # nobody is there, and is stopped the moment someone is.
        self.summarise_careful = summarise_careful
        self.background = background
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def _debug(self, what: str):
        if self.debug:
            self.log.append(("debug", what))

    # -- archive ---------------------------------------------------------------
    def _close(self, enc: Optional[Encounter]):
        """An encounter is over.  Anonymous ones go at once; named ones wait for
        a moment when nobody is there (consolidate), because writing them up
        takes a model call that would otherwise delay the first reply to
        whoever has just arrived."""
        if enc is None:
            return
        if enc.person_id is None:
            self.log.append(("discarded", f"not remembered: {enc.person_name}" if enc.person_name
                             else "anonymous encounter", len(enc.turns)))
            return
        with self._lock:
            self.pending.append(enc)
        self.log.append(("queued", enc.person_name))

    def consolidate(self, person_id: Optional[int] = None):
        """Archive the queued encounters now, with the fast write-up - all of
        them, or one person's.  Used when someone is waiting for it (they are
        back) or at shutdown; a background write-up is stopped first."""
        self._halt_background()
        with self._lock:
            mine = [e for e in self.pending if person_id is None or e.person_id == person_id]
            self.pending = [e for e in self.pending if all(e is not m for m in mine)]
        for enc in mine:
            self._archive(enc)
        return len(mine)

    def _idle(self):
        """Nobody there: time to write up.  In the background with the careful
        write-up if there is one, otherwise right here with the fast one."""
        if not (self.background and self.summarise_careful):
            self.consolidate()
            return
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._careful_worker, daemon=True)
        self._worker.start()

    def _halt_background(self):
        if self._worker is not None and self._worker.is_alive():
            self._stop.set()
            self._worker.join()

    def _careful_worker(self):
        """The careful write-up: thinking on, and a prompt that asks, fact by
        fact, whether what they wrote made an old one untrue.  It catches far
        more outdated facts than the fast one but takes seconds, and the model
        serves one request at a time, so a reply would wait behind it.  It runs
        only while nobody is there and stops the moment someone is; an
        encounter it did not finish stays queued for the next pause."""
        store = type(self.store)(self.store.path)   # sqlite: one connection per thread
        try:
            while not self._stop.is_set():
                with self._lock:
                    if not self.pending:
                        return
                    enc = self.pending[0]
                if not self._archive(enc, store=store, careful=True):
                    return                     # stopped: still queued
                with self._lock:
                    self.pending = [e for e in self.pending if e is not enc]
        finally:
            store.close()

    def _archive(self, enc: Encounter, store=None, careful: bool = False) -> bool:
        store = store or self.store
        old_notes = store.notes(enc.person_id, enc.memory_key)
        turns = [(without(w, TRANSIENT), r) for w, r in enc.turns]
        if careful:
            out = self.summarise_careful(enc.person_name, old_notes, turns, self._stop)
            if out is None:
                return False
            summary, notes = out
        else:
            summary, notes = self.summarise(enc.person_name, old_notes, turns)
        notes = for_archive(self._clean_notes(notes, enc.person_name), self.clock())
        store.add_episode(enc.person_id, enc.started_at, enc.last_face_at, summary, enc.memory_key)
        store.update_notes(enc.person_id, notes, enc.memory_key)
        self._debug(f"{enc.person_name}'s notes were {old_notes!r}, now {notes!r}")
        self.log.append(("archived carefully" if careful else "archived",
                         enc.person_name, summary))
        return True

    def _clean_notes(self, notes: str, own_name: str) -> str:
        """Enforce in code what the prompt asks for and the model does not always do.

        Notes like "She asked about Alessandro's work" carry a question, which the
        prompt forbids, and another person's name.  Sentences with a question,
        or naming anyone else the robot remembers, are dropped.
        """
        own = {w.casefold() for w in own_name.split()}
        knows = self.faces.knows_name if self.faces is not None else (lambda n: False)
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", (notes or "").strip()):
            low = sentence.lower()
            if not sentence or "?" in sentence or re.search(r"\basked\b", low):
                continue
            if any(w.casefold() not in own and knows(w)
                   for w in re.findall(r"\b[A-Z][a-zA-Z\-]+", sentence)):   # "Alessandro's": Alessandro
                continue
            kept.append(sentence)
        return " ".join(kept)

    def _maybe_switch(self, enc: Encounter, t: float):
        """A different name, same face, no pause: the last introduction wins.

        Otherwise Maria taking the board from Alessandro would get what he wrote
        in the same encounter.  Closing and reopening costs, at worst, one
        short extra encounter if it was the same person joking.  A face the
        robot remembers is not switched by a name: the face says who it is.
        """
        new = (enc.record.get("person_name") or "").strip()
        if not enc.person_name or not new or name_key(new) == name_key(enc.person_name):
            return enc, False
        if enc.face_id:
            enc.record["person_name"] = enc.person_name
            self.log.append(("name ignored", f"{new}: the face is {enc.person_name}'s"))
            enc.name_ignored = True
            return enc, False
        old = self.tracker.split(t)
        self._close(old)
        fresh = self.tracker.active
        fresh.record["person_name"] = new
        # The board that caused the switch has been answered; it is not new
        # to the next encounter just because the encounter is new.
        fresh.last_writing, fresh.last_writing_at = old.last_writing, old.last_writing_at
        self.log.append(("switched by name", f"{old.person_name} -> {new}"))
        return fresh, True

    def _maybe_identify(self, enc: Encounter, seen, act, switched=False):
        """A name arrives in the working record: greet them, and ask whether
        their face may be remembered."""
        name = (enc.record.get("person_name") or "").strip()
        if not name or enc.person_name or enc.face_id:
            return act
        enc.person_name = name
        self.log.append(("introduced", name))
        if self.reply_fresh and act.get("action") == "speak":
            text = self.reply_fresh(seen, enc.record)
            if text:
                act = dict(act, text=text)
        if self.faces is not None and not enc.consent:
            enc.consent = "asked"
            text = f"{act.get('text') or ''} {CONSENT_QUESTION}".strip()
            act = dict(act, action="speak", text=text, say=True)
        return act

    # -- faces -------------------------------------------------------------------
    def _recognise(self, enc: Encounter):
        """Two frontal frames that open someone's lock: that person."""
        if self.faces is None or enc.face_id or len(enc.frames) < 2:
            return
        found = self.faces.recognise(enc.frames[-2:])
        if found is not None:
            self._debug(f"face opened {found.name}'s lock (id {found.id}), "
                        f"{found.errors} of 56 face bits corrected")
            self._known(enc, found)
            enc.to_greet = True
        elif len(enc.frames) == 2:
            self._debug(f"face opens none of the {len(self.faces)} remembered")

    def _known(self, enc: Encounter, found):
        """This encounter is with `found` (face_key's Person): their memory,
        opened with their key."""
        enc.face_id, enc.memory_key, enc.consent = found.id, found.memory_key, "yes"
        enc.person_id, created = self.store.person(found.id)
        enc.person_name = enc.record["person_name"] = found.name
        if created:
            self.log.append(("remembered", found.name))
            return
        # Back before their last encounter was written up: write it now, or
        # they would be greeted without what they told the robot minutes ago.
        if self.consolidate(enc.person_id):
            self.log.append(("archived early", found.name))
        enc.memory = self.store.context_for(enc.person_id, found.name, found.memory_key,
                                            self.budget, now=self.clock())
        # What is known about them also goes into the record, where the answer
        # reads facts about the person in front of it.  From the memory block
        # alone, "WHAT AM I BUILDING?" was answered "I am building a robot arm".
        archived = self.store.notes(enc.person_id, found.memory_key).strip()
        current = (enc.record.get("person_notes") or "").strip()
        if archived and archived not in current:
            enc.record["person_notes"] = f"{archived} {current}".strip()
        self.log.append(("recognised by face", found.name))
        self._debug(f"{found.name}'s memory for the prompt: {enc.memory!r}")

    def _intercept(self, enc: Encounter):
        """The boards the models do not answer: the reply to the consent
        question, and asking to be forgotten."""
        def say(what, **fill):
            return {"action": "speak", "text": SAY[what].format(**fill), "say": True,
                    "intercepted": what}

        def take(seen):
            writing = (seen.get("writing") or "").strip()
            if FORGET.search(writing):
                return say(self._forget(enc))
            wanted = UPDATE_NAME.search(writing)
            if wanted:
                what, name = self._ask_rename(enc, " ".join(wanted["name"].split()).title())
                return say(what, name=name)
            if enc.rename_to:
                answer = self._rename(enc, writing)
                return answer and say(answer[0], **answer[1])
            if enc.consent != "asked" or not writing:
                return None
            if NO.search(writing):
                enc.consent = "no"
                self.log.append(("declined", enc.person_name))
                return say("declined")
            if not YES.search(writing):
                # The answer is the next board, or there is none: a YES written
                # later may be the answer to something else the robot asked.
                enc.consent = "no"
                self.log.append(("not answered", enc.person_name))
                return None
            if len(enc.frames) < 2:
                return say("look")
            try:
                found = self.faces.enrol(enc.frames[-4:], enc.person_name)
            except Exception as e:           # the chip busy or gone: nothing kept
                self.log.append(("face error", str(e)))
                return say("failed")
            self._known(enc, found)
            self._debug(f"enrolled {found.name}: id {found.id}")
            return say("already" if found.known else "remembered", name=found.name)
        return take

    def _ask_rename(self, enc: Encounter, name: str):
        """UPDATE MY NAME TO X: asked back first, the name as it was read, so
        that a misread name does not become theirs."""
        if not enc.face_id:
            return "no name to update", ""
        if name_key(name) == name_key(enc.person_name):
            return "same name", enc.person_name
        enc.rename_to = name
        self._debug(f"{enc.person_name} asks to be called {name}")
        return "rename?", name

    def _rename(self, enc: Encounter, writing: str):
        """The answer to "Should I call you X?": the next board, as for consent."""
        new, enc.rename_to = enc.rename_to, ""
        if NO.search(writing):
            return "kept name", {"name": enc.person_name}
        if not YES.search(writing):
            return None                   # not an answer: nothing renamed
        old = enc.person_name
        self.faces.rename(enc.face_id, new)
        self.store.rename(enc.person_id, old, new, enc.memory_key)
        enc.person_name = enc.record["person_name"] = new
        enc.record["person_notes"] = re.sub(rf"\b{re.escape(old)}\b", new,
                                            enc.record.get("person_notes") or "", flags=re.I)
        enc.memory = self.store.context_for(enc.person_id, new, enc.memory_key, self.budget,
                                            now=self.clock())
        self.log.append(("renamed", f"{old} -> {new}"))
        return "renamed", {"name": new}

    def _forget(self, enc: Encounter) -> str:
        if not enc.face_id:
            enc.consent = "no"
            return "nothing"
        self.faces.forget(enc.face_id)
        self.store.forget(enc.face_id)
        with self._lock:
            self.pending = [e for e in self.pending if e.face_id != enc.face_id]
        self.log.append(("forgotten", enc.person_name))
        enc.face_id, enc.memory_key, enc.person_id, enc.memory = "", b"", None, ""
        enc.record["person_notes"], enc.consent = "", "no"
        enc.turns.clear()
        return "forgotten"

    # -- repetition ---------------------------------------------------------------
    @staticmethod
    def _norm(writing):
        return re.sub(r"[^A-Z0-9]", "", (writing or "").upper())

    def _is_new(self, enc: Encounter, t: float):
        """Answer only what is new.

        A board is the same board if it reads the same (ignoring spaces and
        punctuation, or 95% alike on a long text) and was seen within
        repeat_window_s.  Deliberately strict: looser, "WHAT IS YOUR NAME?"
        after "WHAT IS MY NAME?" counts as the same board and is never
        answered, and answering twice costs far less.  Shown again after a
        longer absence, it is answered again.  With nothing written, the robot
        speaks once, and again only after prompt_again_s of saying nothing.
        """
        def is_new(seen):
            w = self._norm(seen.get("writing"))
            if w:
                last = enc.last_writing
                same = bool(last) and t - enc.last_writing_at <= self.repeat_window_s and (
                    w == last or difflib.SequenceMatcher(None, w, last).ratio() >= 0.95)
                enc.last_writing_at = t
                if not same:
                    enc.last_writing = w
                return not same
            return enc.last_spoke_at is None or t - enc.last_spoke_at >= self.prompt_again_s
        return is_new

    def _confirm_name(self, enc: Encounter, before: dict, seen, act, t: float):
        """A new name counts only once the eye has read it on two frames.

        The eye sometimes "reads" an introduction that is not on the board, on
        a single frame.  That would create a person and, by the name-switch
        rule, close the real encounter.  A board really held up is read again a
        second later.  Until then: nothing said, nothing kept.
        Returns (act, confirmed_now).
        """
        if not self.confirm_names:
            return act, False
        writing = seen.get("writing", "") or ""
        pend = enc.pending_name
        if pend and t - enc.pending_at > self.confirm_window_s:
            enc.pending_name = pend = ""
        new = (enc.record.get("person_name") or "").strip()
        if pend and self.confirm_names(writing, pend):
            if name_key(new) != name_key(pend):
                enc.record = enc.pending_record
            enc.pending_name = ""
            return {"action": "speak", "text": "", "say": True}, True
        held = enc.person_name or before.get("person_name", "")
        if new and name_key(new) != name_key(held):
            enc.pending_name, enc.pending_at = new, t
            enc.pending_record = dict(enc.record)
            enc.record = before
            return {"action": "none", "confirming": new}, False
        return act, False

    # -- per frame ----------------------------------------------------------------
    def on_frame(self, face: bool, image_b64, face_vec=None):
        t = self.clock()
        self.last_seen = {}
        if face:
            self._stop.set()          # someone is there: the model is theirs
        event, finished = self.tracker.observe(t, face, face_vec)
        if event in ("ended", "switched"):
            self._close(finished)
        if event == "hold":
            return {"action": "none", "hold": True}
        enc = self.tracker.active
        if enc is None and self.pending:
            self._idle()                       # nobody there: time to write up
        if enc is None or not face:
            return {"action": "none"}
        if face_vec is not None:
            enc.frames = (enc.frames + [face_vec])[-6:]
        self._recognise(enc)
        if self.faces is not None and face_vec is not None and len(enc.frames) == 1:
            # A face just arrived: one frame more and a known face is known.
            # Answered at once, a returning person was told "You're new here".
            return {"action": "none", "looking": True}
        # What John said a moment ago, and to what: a board read half, then whole,
        # is answered as one (two_stage.continuing).
        last = enc.turns[-1] if enc.turns else None
        recent = enc.last_spoke_at is not None and t - enc.last_spoke_at <= self.continue_window_s
        enc.record["_last_turn"] = (tuple(last) if last and last[0] and recent
                                    and last[1] not in ("", "none") else None)
        before = dict(enc.record)
        extra = {"intercept": self._intercept(enc)} if self.faces is not None else {}
        act, seen = self.decide(image_b64, enc.record, enc.memory, self._is_new(enc, t), **extra)
        self.last_seen = seen
        if act.get("unfinished"):
            return act                         # half-read: the next frame reads it whole
        if act.get("chess"):
            # A game belongs to the table, not to this encounter: none of it is
            # kept - no turns, no names, nothing for the write-up.
            if act.get("action") == "speak":
                enc.last_spoke_at = t
            return act
        if act.get("intercepted"):
            enc.last_spoke_at = t
            return act                         # not part of the conversation
        act, confirmed = self._confirm_name(enc, before, seen, act, t)
        if act.get("confirming"):
            return act                         # read once: wait for the next frame
        enc, switched = self._maybe_switch(enc, t)
        act = self._maybe_identify(enc, seen, act, switched)
        if enc.name_ignored:
            # Not them, or misread: nothing of that board is kept - it would
            # have made "Maria is building a robot arm" of Alessandro's notes.
            enc.name_ignored = False
            enc.record = dict(before, person_name=enc.person_name)
            enc.last_spoke_at = t
            return {"action": "speak", "text": SAY["known as"].format(name=enc.person_name),
                    "say": True}
        if enc.to_greet and act.get("action") in ("speak", "none") and self.reply_with_memory:
            # Recognised on this frame.  A board was answered already knowing
            # who they are; with nothing written, they are greeted by name.
            # Both, and "HELLO" got two greetings.
            enc.to_greet = False
            if not seen.get("writing"):
                act = {"action": "speak", "say": True,
                       "text": self.reply_with_memory(seen, enc.record, enc.memory)}
        # The working record gets the same cleaning as the archive, or Maria
        # asking "WHAT IS ALESSANDRO BUILDING?" leaves Alessandro in her notes and in
        # the next replies to her.
        enc.record["person_notes"] = self._clean_notes(
            enc.record.get("person_notes", ""), enc.person_name)
        if act.get("action") in ("speak", "gesture"):
            enc.last_spoke_at = t
        if act.get("repeat") and not confirmed:
            return act
        enc.turns.append((seen.get("writing", ""),
                          act.get("text") or act.get("gesture") or act.get("action", "")))
        return act

    def shutdown(self):
        self._close(self.tracker.flush())
        self.consolidate()
