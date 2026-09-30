#!/usr/bin/env python3
"""The robot's live loop, running on the GPU server.

Every --interval seconds it reads /status and a snapshot from the robot bridge
on the Pi.  A gate with no model in it decides whether anything is worth
looking at (a face, or a scene that changed), so an empty room costs no
inference.  Each frame goes to Mind (encounters.py), which tracks who the
robot is talking to and asks two_stage.step() what to do.  The result is said
aloud (/speak) or made with the hand (/arm).

    python3 one_mind.py --dry-run --max-seconds 60   # decide, send nothing
    python3 one_mind.py --face-check                 # live

People are remembered only with --face-check, by face and with their consent
(encounters.py), and only once an admin key exists (pi/face_key/admin.py
keygen, public key in memory/admin.pub).

    python3 one_mind.py --dry-run --face-check --debug   # everything shown
    python3 inspect_memory.py                            # everything kept, decrypted

--debug uses a memory of its own (memory/debug/), writes every key there in
the clear, and logs what is otherwise only kept encrypted.  Not for real
people's memory.

See the README for the whole picture.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from dataclasses import dataclass

import two_stage
from encounters import EncounterTracker, Mind
from memory_store import MemoryStore
from pathlib import Path

from paths import DEBUG_MEMORY, FACE_KEYS, LIVE_LOGS, MEMORY_DB  # noqa: E402

OLLAMA = "http://127.0.0.1:11434/api/chat"
BRIDGE = "http://192.168.50.2:8088"


# --------------------------------------------------------------------------
# The gate.  No model runs unless this opens.
# --------------------------------------------------------------------------
@dataclass
class Gate:
    """Decides whether the scene is worth a model call.

    It opens when a face is present, or when the picture changed a lot since
    the last look.  The second catches a whiteboard held up without a detected
    face.
    """

    # Above the camera's frame-to-frame noise on an empty room, below a person
    # or a whiteboard appearing.  The margin is thin: this is a coarse secondary
    # trigger, and the face flag carries the load.
    change_threshold: float = 40.0   # mean abs difference, 0-255, on a tiny image
    _last_thumb: object = None
    opened: int = 0
    closed: int = 0

    def _thumb(self, jpeg: bytes):
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        return None if img is None else cv2.resize(img, (32, 24)).astype("float32")

    def check(self, face_found: bool, jpeg: bytes | None):
        reason = ""
        if face_found:
            reason = "face present"
        thumb = self._thumb(jpeg) if jpeg else None
        if thumb is not None:
            if self._last_thumb is not None:
                import numpy as np
                diff = float(np.abs(thumb - self._last_thumb).mean())
                if not reason and diff >= self.change_threshold:
                    reason = f"scene changed ({diff:.0f})"
            self._last_thumb = thumb
        if reason:
            self.opened += 1
        else:
            self.closed += 1
        return reason

    @staticmethod
    def is_face(reason: str) -> bool:
        return reason.startswith("face")


def snapshot():
    with urllib.request.urlopen(f"{BRIDGE}/snapshot.jpg?w=480&q=85", timeout=10) as r:
        return r.read()


def status():
    with urllib.request.urlopen(f"{BRIDGE}/status", timeout=5) as r:
        return json.load(r)


def send(action, dry):
    if dry or action.get("action") in (None, "none"):
        return
    if action.get("action") == "speak":
        # Always aloud: the model sometimes emits "say": false and would go silent.
        body = {"text": action.get("text", ""), "say": True}
        url = f"{BRIDGE}/speak"
    elif action.get("action") == "gesture":
        body = {"action": "gesture", "gesture": action.get("gesture", "")}
        url = f"{BRIDGE}/arm"
    else:
        return
    posts = [(url, body)]
    if action.get("action") == "speak" and action.get("gesture"):
        # The reasoning stage chose a movement to go with what it says.
        posts.append((f"{BRIDGE}/arm", {"action": "gesture", "gesture": action["gesture"]}))
    for url, body in posts:
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as e:
            print(f"  [warn] send failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="decide but send nothing")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=1.0, help="gate polling period")
    ap.add_argument("--memory", default=str(MEMORY_DB),
                    help="archive of people who introduced themselves (SQLite)")
    ap.add_argument("--encounter-timeout", type=float, default=5.0,
                    help="seconds with nobody there before an encounter is over")
    ap.add_argument("--face-check", action="store_true",
                    help="compare faces to tell a new person from the same one "
                         "(only after validate_faces.py on this camera)")
    ap.add_argument("--face-threshold", type=float, default=0.363)
    ap.add_argument("--debug", action="store_true",
                    help="a separate memory (memory/debug/) with its keys in the clear, and "
                         "every step logged: for testing, never with real people's memory")
    ap.add_argument("--chip", default="",
                    help="with --face-check: serial port of the key chip (e.g. /dev/ttyACM0); "
                         "without it a software stand-in, for development only")
    ap.add_argument("--log", default=str(LIVE_LOGS
                                         / time.strftime("%Y%m%d_%H%M%S.jsonl")),
                    help="every turn of the session, for looking at it afterwards")
    ap.add_argument("--reappear-seconds", type=float, default=90.0,
                    help="with --face-check: how long someone can be away (rubbing "
                         "out the board, writing) and still be the same encounter")
    a = ap.parse_args()
    if a.debug:
        if a.memory == str(MEMORY_DB):
            a.memory = str(DEBUG_MEMORY / "memory.sqlite")
        print(f"[debug] memory {a.memory}: keys written in the clear next to it")
        if Path(a.memory).parent.resolve() == MEMORY_DB.parent.resolve():
            print("[debug] WARNING: this is the real memory; its protection is being undone")

    gate = Gate()
    faces = keys = None
    if a.face_check:
        from face_compare import FaceComparer
        faces = FaceComparer(threshold=a.face_threshold)
        keys = face_keys(a.chip, Path(a.memory).parent, a.debug)
    store = MemoryStore(a.memory)
    started = time.time()
    mind = Mind(
        store, EncounterTracker(timeout_s=a.encounter_timeout,
                                same_person=faces.same if faces else None,
                                reappear_s=a.reappear_seconds),
        decide=lambda img, rec, mem, is_new, **kw: two_stage.step(
            img, rec, keep_record=True, memory=mem,
            self_memory=store.self_context(started_at=started), is_new=is_new,
            others=keys.knows_name if keys is not None else (), **kw),
        summarise=two_stage.summarise_encounter,
        reply_with_memory=lambda seen, rec, mem: two_stage.greet_returning(rec, mem),
        reply_fresh=lambda seen, rec: two_stage.greet_new(rec),
        confirm_names=two_stage.is_self_introduction,
        summarise_careful=(lambda n, o, t, stop: two_stage.summarise_encounter(
            n, o, t, careful=True, stop=stop)) if two_stage.CAREFUL_SUMMARY else None,
        background=True, faces=keys, debug=a.debug)
    print("loading models into VRAM...", flush=True)
    print(f"ready in {two_stage.warmup():.1f}s\n", flush=True)

    print(f"vision={two_stage.VISION_MODEL}  reasoning={two_stage.TEXT_MODEL}  "
          f"dry_run={a.dry_run}  memory={a.memory}  "
          f"faces={'compare, threshold ' + str(a.face_threshold) if faces else 'no'}  "
          f"remembers={'faces, ' + str(len(keys)) + ' people' if keys is not None else 'nobody'}\n")
    t0 = time.time()
    try:
        _loop(a, gate, mind, t0, faces)
    finally:
        # An encounter still open at exit is archived, not lost.
        before = len(mind.log)
        mind.shutdown()
        for ev in mind.log[before:]:
            print(f"  [memory] {ev[0]}: {ev[1]}")

    total = gate.opened + gate.closed
    saved = 100.0 * gate.closed / total if total else 0.0
    print(f"\ncycles {total}   gate open {gate.opened}   closed {gate.closed} "
          f"({saved:.0f}% inferences saved)")
    print(f"people remembered: {store.people()}")


def face_keys(port: str, where: Path, debug: bool = False):
    """Faces as keys (pi/face_key), kept in `where` next to the memory, or
    None - and then nobody is remembered - if there is no admin key there to
    seal names for.  In debug, a test admin key is made there if missing,
    and every key is written to keys.json."""
    sys.path.insert(0, str(FACE_KEYS))
    from face_key.chip import SerialChip, SoftChip
    from face_key.store import FaceKeys, admin_keypair
    admin = where / "admin.pub"
    if debug and not admin.exists():
        private, public = admin_keypair()
        where.mkdir(parents=True, exist_ok=True)
        (where / "admin.key").write_text(private.hex() + "\n")
        (where / "admin.key").chmod(0o600)
        admin.write_text(public.hex() + "\n")
        print(f"[debug] test admin key made: {where / 'admin.key'}")
    if not admin.exists():
        print(f"[faces] no admin key at {admin}: nobody will be remembered.\n"
              f"        Make one with: python3 -m face_key.admin keygen <file off the robot>")
        return None
    chip = SerialChip(port) if port else SoftChip(where / "soft_chip.key")
    if not port:
        print("[faces] software key chip: for development only")
    return FaceKeys(chip, bytes.fromhex(admin.read_text().strip()), where / "faces.sqlite",
                    escrow=where / "keys.json" if debug else None)


def _loop(a, gate, mind, t0, faces=None):
    n_log = 0
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    log = open(a.log, "a")
    print(f"session log: {a.log}\n")
    while time.time() - t0 < a.max_seconds:
        loop_start = time.time()
        try:
            st = status()
            jpeg = snapshot()
        except Exception as e:
            print(f"[warn] bridge: {e}")
            time.sleep(1.0)
            continue

        reason = gate.check(bool(st.get("face", {}).get("found")), jpeg)
        # Every cycle goes to Mind, open gate or not: a closed gate tells it
        # nobody is there, which is how an encounter ends.  No model runs then.
        img = base64.b64encode(jpeg).decode() if reason else ""
        # The face vector stays in RAM, inside the encounter; never saved.
        vec = faces.embed(jpeg) if (faces and reason) else None
        act = mind.on_frame(bool(reason), img, vec)
        events = mind.log[n_log:]
        for ev in events:
            print(f"  [memory] {ev[0]}: {ev[1]}", flush=True)
        n_log = len(mind.log)
        if reason or events:
            entry = {"t": round(time.time() - t0, 1), "reason": reason,
                     "writing": (mind.last_seen or {}).get("writing", ""),
                     "act": act, "memory": [list(map(str, e)) for e in events]}
            enc = mind.tracker.active
            if a.debug and enc is not None:
                entry["debug"] = {"person": enc.person_name, "face": enc.face_id,
                                  "consent": enc.consent, "record": enc.record,
                                  "memory": enc.memory, "frames": len(enc.frames)}
            log.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            log.flush()
        if not reason:
            time.sleep(max(0.0, a.interval - (time.time() - loop_start)))
            continue
        seen = mind.last_seen or {}
        extra = f"  [gesture: {act['gesture']}]" if act.get("gesture") and act.get("text") else ""
        txt = (act.get("text") or act.get("gesture") or (
            "(already answered)" if act.get("repeat") else
            "(waiting for a frontal face)" if act.get("hold") else
            "(looking at a new face)" if act.get("looking") else
            f"(name {act['confirming']!r} read once: waiting for confirmation)" if act.get("confirming")
            else act.get("action", "?"))) + extra
        wrote = " / ".join(str(seen.get("writing", "")).split("\n"))
        print(f"[{time.time() - t0:5.1f}s] {reason:20} -> {txt}"
              + (f"\n          writing: {wrote!r}" if wrote else ""), flush=True)
        send(act, a.dry_run)
        time.sleep(max(0.0, a.interval - (time.time() - loop_start)))


if __name__ == "__main__":
    main()
