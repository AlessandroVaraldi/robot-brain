"""The secret that stays on the robot: HMAC-SHA256 under a key nobody can read.

On the robot this is an ESP32-C3 on USB, with the key in a read-protected
eFuse used by its HMAC peripheral: the Pi can ask for MACs, never for the key.
Without the chip, what is stored on the Pi opens nothing.

SoftChip does the same in software, with the key in a file, for development
and tests. It protects nothing.

What the chip is asked (the label is part of the message, and nothing else is
answered):
  locks    once, at start: the key the locks are stored under
  check    with a key from a lock: whether it is the right one
  memory   with a key from a lock: the key to that person's memory
  names    once, at start: the key names are tagged with, so that the brain
           can tell whether it knows a name without keeping names
The data is at most 64 bytes. Every answer costs a token; tokens come back at
one a second, up to 20, so the chip cannot be used to try keys quickly.

SerialChip talks to the chip (firmware/) over USB; SoftChip is the same in
software.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from pathlib import Path

LABELS = (b"locks", b"check", b"memory", b"names")
DATA_MAX = 64


class ChipBusy(Exception):
    """Asked too often: try again later."""


class ChipError(Exception):
    """No chip, not John's, no key in it, or a request it refused."""


class SerialChip:
    """The ESP32-C3 on USB, or anything that speaks its protocol on `stream`
    (a line out, a line back; see firmware/main/chip.c)."""

    def __init__(self, port: str = "/dev/ttyACM0", stream=None):
        if stream is None:
            import serial
            stream = serial.Serial(port, 115200, timeout=2)
        self.io = stream
        hello = self._ask("HELLO")
        if hello == "JOHN-CHIP 1 NOKEY":
            raise ChipError("the chip has no key burnt in it")
        if hello != "JOHN-CHIP 1 KEY":
            raise ChipError(f"not John's chip: {hello!r}")

    def _ask(self, line: str) -> str:
        self.io.write(line.encode() + b"\n")
        self.io.flush()
        got = self.io.readline()
        if not got.endswith(b"\n"):
            raise ChipError("no answer from the chip")
        return got.decode().strip()

    def mac(self, label: bytes, data: bytes = b"") -> bytes:
        if label not in LABELS or len(data) > DATA_MAX:
            raise ValueError(f"the chip does not answer {label!r} with {len(data)} bytes")
        answer = self._ask(f"MAC {label.decode()} {data.hex()}")
        if answer == "BUSY":
            raise ChipBusy
        if answer.startswith("OK "):
            return bytes.fromhex(answer[3:])
        raise ChipError(answer)


class SoftChip:
    def __init__(self, key_file: Path | None = None, key: bytes | None = None,
                 per_minute: float = 60, burst: int = 20, clock=time.monotonic):
        if key is None:
            key_file = Path(key_file)
            if not key_file.exists():
                key_file.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.write(fd, os.urandom(32))
                os.close(fd)
            key = key_file.read_bytes()
        self._key = key
        self._rate = per_minute / 60
        self._burst = self._tokens = float(burst)
        self._clock = clock
        self._t = clock()

    def mac(self, label: bytes, data: bytes = b"") -> bytes:
        if label not in LABELS or len(data) > DATA_MAX:
            raise ValueError(f"the chip does not answer {label!r} with {len(data)} bytes")
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._t) * self._rate)
        self._t = now
        if self._tokens < 1:
            raise ChipBusy
        self._tokens -= 1
        return hmac.new(self._key, label + b":" + data, hashlib.sha256).digest()
