#!/usr/bin/env python3
"""Run the live loop against a fake bridge, with the robot off.

Exercises the whole loop - gate, two stages, record kept across turns - without
the robot.  A local server answers /status and /snapshot.jpg from saved frames
in a scripted order that plays out an encounter: an empty lab, someone walking
up and introducing themselves, asking about themselves and about the robot,
leaving, then coming back.

Nothing here talks to the robot; one_mind runs with --dry-run.
"""

import base64
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "john"))
import one_mind  # noqa: E402
from fixtures import board_on_person  # noqa: E402

from paths import FRAMES  # noqa: E402

def jpeg_of(path):
    return Path(path).read_bytes()


def board_jpeg(text):
    return base64.b64decode(board_on_person(text))


# (seconds on screen, face flag the real bridge would report, label, jpeg bytes)
SCRIPT = [
    (4, False, "empty room",               jpeg_of(FRAMES / "scene.jpg")),
    (4, True,  "person looking",           jpeg_of(FRAMES / "pose1.jpg")),
    (4, True,  "MY NAME IS ALESSANDRO",       board_jpeg("MY NAME IS ALESSANDRO")),
    (4, True,  "YES",                      board_jpeg("YES")),
    (4, True,  "I AM BUILDING A ROBOT ARM", board_jpeg("I AM BUILDING A ROBOT ARM")),
    (4, True,  "WHAT IS MY NAME?",         board_jpeg("WHAT IS MY NAME?")),
    (4, True,  "WHAT IS YOUR NAME?",       board_jpeg("WHAT IS YOUR NAME?")),
    (4, False, "looks at the phone",       jpeg_of(FRAMES / "pose2.jpg")),
    (8, False, "empty room",               jpeg_of(FRAMES / "quiet2.jpg")),
    # He comes back: with --face-check, recognised by his face.
    (4, True,  "person looking",           jpeg_of(FRAMES / "pose1.jpg")),
    (4, True,  "WHAT AM I BUILDING?",      board_jpeg("WHAT AM I BUILDING?")),
    (4, True,  "UPDATE MY NAME TO LUCA",   board_jpeg("UPDATE MY NAME TO LUCA")),
    (4, True,  "YES",                      board_jpeg("YES")),
    (4, True,  "WHAT IS MY NAME?",         board_jpeg("WHAT IS MY NAME?")),
    (8, False, "empty room",               jpeg_of(FRAMES / "quiet2.jpg")),
]
T0 = time.time()


def current():
    t = time.time() - T0
    for secs, face, label, jpeg in SCRIPT:
        if t < secs:
            return face, label, jpeg
        t -= secs
    return SCRIPT[-1][1], SCRIPT[-1][2], SCRIPT[-1][3]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        face, label, jpeg = current()
        if self.path.startswith("/snapshot.jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(jpeg)
        elif self.path.startswith("/status"):
            body = json.dumps({"face": {"found": face, "ex": 0, "ey": 0},
                               "tracking": {"mode": "off"},
                               "scene_label": label}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


EXTRA = sys.argv[1:]      # e.g. --face-check, passed through to one_mind


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 18088), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    one_mind.BRIDGE = "http://127.0.0.1:18088"
    import two_stage
    global T0
    print(f"warm-up: {two_stage.warmup():.1f}s", flush=True)
    T0 = time.time()      # the encounter starts once the robot is ready
    total = sum(s for s, *_ in SCRIPT)
    print(f"fake bridge on :18088, script of {total}s\n")
    # A throwaway archive: the real one is for real people.
    db = Path(tempfile.mkdtemp()) / "memory.sqlite"
    sys.argv = ["one_mind.py", "--dry-run", "--max-seconds", str(total + 2),
                "--interval", "1.0", "--memory", str(db)] + EXTRA
    one_mind.main()
    server.shutdown()
    if "--debug" in EXTRA:
        from inspect_memory import inspect
        print("\n" + inspect(db.parent))


if __name__ == "__main__":
    main()
