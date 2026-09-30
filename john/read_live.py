#!/usr/bin/env python3
"""Print the turns of a live session log (one_mind.py --log): what was written,
what was decided, what memory did.  Default: the latest log."""
import glob, json, os, sys
from paths import LIVE_LOGS as d  # noqa: E402
f = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(str(d / "*.jsonl")), key=os.path.getmtime)
print("file:", f)
for line in open(f):
    r = json.loads(line)
    a = r["act"]
    if not (r["writing"] or r["memory"] or a.get("action") != "none"):
        continue
    what = (a.get("text") or a.get("gesture") or ("(already answered)" if a.get("repeat") else
            "(waiting for face)" if a.get("hold") else a.get("action")))
    mem = "  " + "; ".join(" ".join(e[:2]) for e in r["memory"]) if r["memory"] else ""
    print(f"{r['t']:6} {r['writing']!r:44} -> {a.get('action')}: {what}{mem}")
