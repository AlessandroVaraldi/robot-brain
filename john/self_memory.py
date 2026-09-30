#!/usr/bin/env python3
"""What the lab team tells the robot about itself.  Operator only.

This is the only way a fact about the robot gets into its memory: what people
write on the whiteboard about it is answered, never kept.

    python3 self_memory.py add "Your hand was replaced with a new one."
    python3 self_memory.py list
    python3 self_memory.py remove 3
    python3 self_memory.py show        # exactly what the robot is given

Write facts addressed to the robot, in the second person ("Your hand...", "You
will be at the open day on Friday"): they are read as things about itself.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_store import MemoryStore, when  # noqa: E402

from paths import MEMORY_DB  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", default=str(MEMORY_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("text")
    sub.add_parser("list")
    rm = sub.add_parser("remove")
    rm.add_argument("id", type=int)
    sub.add_parser("show")
    a = ap.parse_args()

    store = MemoryStore(a.memory)
    if a.cmd == "add":
        print(f"added #{store.add_self_fact(a.text)}")
    elif a.cmd == "list":
        import time
        now = time.time()
        for f in store.self_facts():
            print(f"#{f['id']:<4} {when(now - f['created_at']):>18}  {f['text']}")
    elif a.cmd == "remove":
        print("removed" if store.remove_self_fact(a.id) else f"#{a.id} does not exist")
    else:
        print(store.self_context())


if __name__ == "__main__":
    main()
