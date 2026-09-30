"""The admin's view of John's face memory: the names, and deleting people by
name (a misread name must not stay attached to a face).

The names are sealed with the admin's public key. The private key never goes
on the robot: keep it on your own machine and hand it over on stdin.

    python3 -m face_key.admin keygen ~/.john/admin.key     # once; prints the public key
    ssh robot-pi 'cd robot-brain/pi && python3 -m face_key.admin list --key -' < ~/.john/admin.key
    ssh robot-pi 'cd robot-brain/pi && python3 -m face_key.admin delete Luca --key -' < ~/.john/admin.key

The public key goes on the robot in memory/admin.pub. `delete` only shows who
would go; add --yes to delete them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .store import STORE, admin_keypair, delete, listing, open_as_admin


def read_key(where: str) -> bytes:
    text = sys.stdin.read() if where == "-" else Path(where).expanduser().read_text()
    return bytes.fromhex(text.strip())


def people(private: bytes, store: Path) -> list[tuple[str, str, float, int]]:
    """(id, name, created, locks) for everyone stored."""
    return [(pid, open_as_admin(private, sealed), created, locks)
            for pid, sealed, created, locks in listing(store)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="face_key.admin", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    kg = sub.add_parser("keygen", help="a new admin key pair")
    kg.add_argument("private_file", help="where to write the private key (not on the robot)")
    for name in ("list", "delete"):
        p = sub.add_parser(name)
        p.add_argument("--key", required=True, help="the private key file, or - for stdin")
        p.add_argument("--store", type=Path, default=STORE)
        if name == "delete":
            p.add_argument("name", help="as listed; case does not matter")
            p.add_argument("--yes", action="store_true", help="delete, do not just show")
    a = ap.parse_args(argv)

    if a.cmd == "keygen":
        out = Path(a.private_file).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        private, public = admin_keypair()
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, private.hex().encode() + b"\n")
        os.close(fd)
        print(public.hex())
        return 0

    everyone = people(read_key(a.key), a.store)
    if a.cmd == "list":
        for pid, name, created, locks in everyone:
            day = time.strftime("%d/%m/%Y", time.localtime(created))
            print(f"{name:24} since {day}  locks {locks}  id {pid}")
        print(f"{len(everyone)} people")
        return 0

    matches = [p for p in everyone if p[1].strip().casefold() == a.name.strip().casefold()]
    if not matches:
        print(f"nobody called {a.name!r}")
        return 1
    for pid, name, _, _ in matches:
        if a.yes:
            delete(pid, a.store)
        print(f"{'deleted' if a.yes else 'would delete'}: {name} (id {pid})")
    if not a.yes:
        print("add --yes to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
