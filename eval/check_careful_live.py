#!/usr/bin/env python3
"""End to end, real models: a careful write-up that finishes during a long enough
pause, on a fact that has stopped being true."""
import tempfile, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import two_stage as T  # noqa: E402
from fixtures import Faces, board_on_person  # noqa: E402
from encounters import EncounterTracker, Mind  # noqa: E402
from memory_store import MemoryStore  # noqa: E402

T.warmup()
faces = Faces()
store = MemoryStore(Path(tempfile.mkdtemp()) / "mem.sqlite")
alessandro = faces.keys.enrol([faces.frame("alessandro"), faces.frame("alessandro")], "Alessandro")
store.update_notes(store.person(alessandro.id)[0], "Alessandro is building a robot arm.", alessandro.memory_key)
clock = [1_760_000_000.0]
mind = Mind(store, EncounterTracker(timeout_s=5.0, same_person=faces.same),
            decide=lambda img, rec, mem, is_new, **kw: T.step(img, rec, keep_record=True, memory=mem,
                                                              is_new=is_new, **kw),
            summarise=T.summarise_encounter,
            reply_with_memory=lambda s, r, m: T.greet_returning(r, m),
            reply_fresh=lambda s, r: T.greet_new(r), clock=lambda: clock[0],
            confirm_names=T.is_self_introduction,
            summarise_careful=lambda n, o, t, stop: T.summarise_encounter(n, o, t, careful=True, stop=stop),
            background=True, faces=faces.keys)
for board in ("HELLO", "HELLO", "I FINISHED THE ARM. NOW I AM BUILDING A DRONE"):
    clock[0] += 2
    act = mind.on_frame(True, board_on_person(board), faces.frame("alessandro"))
    print(f"{board:48} -> {act.get('text') or act.get('action')}")
t0 = time.time()
while time.time() - t0 < 90:                      # the room stays empty
    clock[0] += 1
    mind.on_frame(False, "")
    if any(e[0] == "archived carefully" for e in mind.log):
        break
    time.sleep(0.5)
print(f"\ncareful write-up finished in {time.time() - t0:.1f}s:", [e[0] for e in mind.log][-3:])
print("notes in archive:", repr(store.notes(store.person(alessandro.id)[0], alessandro.memory_key)))
