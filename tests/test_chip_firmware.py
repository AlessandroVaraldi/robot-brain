#!/usr/bin/env python3
"""The chip's firmware (pi/face_key/firmware/main/chip.c), built for this
computer with stand-ins for ESP-IDF (tests/chip_host/: the USB port is stdin
and stdout, the eFuse key bytes 0..31), and spoken to by SerialChip as the
robot would: its answers must be SoftChip's, its refusals the same. Needs a
C compiler; no chip, no ESP-IDF."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pi"))
from face_key.chip import DATA_MAX, LABELS, ChipBusy, ChipError, SerialChip, SoftChip  # noqa: E402
from face_key.store import FaceKeys, admin_keypair  # noqa: E402

HOST = ROOT / "tests" / "chip_host"
SOURCES = [str(HOST / "host.c"), str(ROOT / "pi" / "face_key" / "firmware" / "main" / "chip.c")]
KEY = bytes(range(32))
RESULTS = []


def check(ok, label):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


def build(out: Path) -> str:
    """"" if built, else the compiler's complaint."""
    flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(HOST), "-o", str(out)]
    tries = [[]]
    sdks = Path("/Library/Developer/CommandLineTools/SDKs")
    if sys.platform == "darwin" and sdks.exists():
        # a linker older than the newest SDK cannot link against it
        tries += [["-isysroot", str(s)] for s in sorted(sdks.glob("MacOSX*.*.sdk"), reverse=True)]
    complaint = "no C compiler"
    for extra in tries:
        try:
            done = subprocess.run(["cc", *extra, *flags, *SOURCES], capture_output=True, text=True)
        except FileNotFoundError:
            break
        if done.returncode == 0:
            return ""
        complaint = done.stderr
    return complaint


class Pipe:
    """The chip process as a serial port."""
    def __init__(self, exe, **env):
        self.p = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  bufsize=0, env={**os.environ, **env})

    def write(self, b):
        self.p.stdin.write(b)

    def flush(self):
        self.p.stdin.flush()

    def readline(self):
        return self.p.stdout.readline()

    def ask(self, line):
        self.write(line.encode() + b"\n")
        return self.readline().decode().strip()

    def close(self):
        self.p.stdin.close()
        self.p.wait(5)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "chip"
        complaint = build(exe)
        if complaint:
            print(f"  cannot build the firmware for this computer:\n{complaint[-1500:]}")
            return 1
        soft = SoftChip(key=KEY, burst=10**6)
        rng = np.random.default_rng(0)

        pipe = Pipe(exe)
        chip = SerialChip(stream=pipe)
        check(True, "HELLO: John's chip, with a key")
        same = all(chip.mac(label, data) == soft.mac(label, data)
                   for label in LABELS
                   for data in (b"", bytes(rng.integers(0, 256, 15, dtype=np.uint8)),
                                bytes(rng.integers(0, 256, DATA_MAX, dtype=np.uint8))))
        check(same, "every label, empty to 64 bytes: the same MACs as SoftChip")
        check(pipe.ask("MAC names " + "00" * (DATA_MAX + 1)) == "ERR bad data", "65 bytes: refused")
        check(pipe.ask("MAC check 0") == "ERR bad data" and pipe.ask("MAC check zz") == "ERR bad data",
              "odd or non-hex data: refused")
        check(pipe.ask("MAC secret 00") == "ERR unknown label", "an unknown label: refused")
        check(pipe.ask("KEY") == "ERR unknown request", "anything else: refused")
        check(pipe.ask("MAC check " + "0" * 400) == "ERR too long" and pipe.ask("HELLO").startswith("JOHN"),
              "a line too long: refused, and the next one read")
        try:
            chip.mac(b"check", bytes(DATA_MAX + 1))
            check(False, "SerialChip does not send what the chip would refuse")
        except ValueError:
            check(True, "SerialChip does not send what the chip would refuse")
        pipe.close()

        pipe = Pipe(exe)
        chip = SerialChip(stream=pipe)
        answered = 0
        try:
            for _ in range(30):
                chip.mac(b"check", b"x")
                answered += 1
        except ChipBusy:
            pass
        check(answered == 20, f"20 answers at once, then busy ({answered})")
        time.sleep(1.1)
        check(chip.mac(b"check", b"x") == soft.mac(b"check", b"x"), "a second later, one more")
        pipe.close()

        pipe = Pipe(exe, CHIP_HOST_NOKEY="1")
        try:
            SerialChip(stream=pipe)
            check(False, "no key burnt: SerialChip says so")
        except ChipError:
            check(True, "no key burnt: SerialChip says so")
        check(pipe.ask("MAC check 00") == "ERR no key", "and the chip answers nothing")
        pipe.close()

        pipe = Pipe(exe)
        keys = FaceKeys(SerialChip(stream=pipe), admin_keypair()[1], Path(tmp) / "faces.sqlite")
        face = rng.standard_normal(128)

        def frame():
            return face + 0.8 * rng.standard_normal(128) / np.sqrt(128)
        p = keys.enrol([frame(), frame()], "Luca")
        got = keys.recognise([frame(), frame()])
        check(got and got.id == p.id and got.name == "Luca", "face keys through the firmware: enrol, recognise")
        pipe.close()

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
