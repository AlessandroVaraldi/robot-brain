#!/usr/bin/env python3
"""Summarise robust.py runs: how much of each run survives a change of wording.

    python3 eval/robust_report.py eval/suite_out/robust_*.json [--cases TAG]

Per case, passes under each of the four wordings (out of n):
  solid    passes >= n-1 under EVERY wording
  broken   passes <= 1 under every wording
  fragile  the spread between wordings is n/2 or more
  other    none of the above
Verdicts are recomputed with the current suite.py.
"""

import json
from pathlib import Path
import re
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import suite  # noqa: E402

SPEC = {(c, x, b): (m, mn, ma) for c, x, b, m, mn, ma in suite.A}


def per_case(run):
    table = defaultdict(dict)
    for w, rows in run["wordings"].items():
        for r in rows:
            key = (r["cat"], r["ctx"], r["board"])
            m, mn, ma = SPEC[key]
            if run.get("clock") and key[2] in suite.WITH_CLOCK:
                m, mn = suite.WITH_CLOCK[key[2]]
            ok = not suite.check(r["out"], m, mn, ma)
            p, n = table[key].get(w, (0, 0))
            table[key][w] = (p + ok, n + 1)
    return table


def classify(scores):
    ps = [p for p, _ in scores.values()]
    n = max(n for _, n in scores.values())
    return ("solid" if min(ps) >= n - 1 else "broken" if max(ps) <= 1 else
            "fragile" if max(ps) - min(ps) >= n / 2 else "other")


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".json")]
    runs = {}
    for p in paths:
        run = json.load(open(p))
        tag = re.search(r"robust_(.+?)_\d{8}_", p).group(1)
        runs[tag] = run
    print(f"{'run':10} {'model':16} {'solid':>7} {'fragile':>8} {'broken':>6} {'mean':>7} "
          f"{'caps':>7} {'empty':>6} {'latency':>8}")
    tables = {}
    for tag, run in runs.items():
        t = tables[tag] = per_case(run)
        kinds = defaultdict(int)
        for scores in t.values():
            kinds[classify(scores)] += 1
        allrows = [r for rows in run["wordings"].values() for r in rows]
        mean = sum(p for s in t.values() for p, _ in s.values()) / sum(
            n for s in t.values() for _, n in s.values())
        caps = sum(1 for r in allrows if r["out"] and r["out"] == r["out"].upper()
                   and re.search("[A-Z]{3}", r["out"]))
        empty = sum(1 for r in allrows if not r["out"].strip())
        print(f"{tag:10} {run['model'] + (' +body' if run['body'] else ''):16} {kinds['solid']:>7} "
              f"{kinds['fragile']:>8} {kinds['broken']:>6} {mean:>6.0%} {caps:>7} {empty:>6} "
              f"{run['latency_ms']:>6.0f}ms")

    print(f"\nper category (mean over the 4 wordings)\n{'':14}" + "".join(f"{t:>9}" for t in runs))
    cats = sorted({k[0] for t in tables.values() for k in t})
    for c in cats:
        cells = []
        for tag, t in tables.items():
            s = [v for k, sc in t.items() if k[0] == c for v in sc.values()]
            # a category a run does not have (the suite grew since): "-"
            cells.append(f"{sum(p for p, _ in s) / sum(n for _, n in s):>8.0%}" if s else f"{'-':>8}")
        print(f"  {c:12}" + " ".join(cells))

    if "--cases" in sys.argv:
        tag = sys.argv[sys.argv.index("--cases") + 1]
        run, t = runs[tag], tables[tag]
        print(f"\ncases that are not solid in {tag} (passes per wording P0 P1 P2 P3)")
        outs = defaultdict(lambda: defaultdict(list))
        for w, rows in run["wordings"].items():
            for r in rows:
                outs[(r["cat"], r["ctx"], r["board"])][w].append(r["out"])
        for key, scores in t.items():
            k = classify(scores)
            if k == "solid":
                continue
            cells = " ".join(f"{scores[w][0]}/{scores[w][1]}" for w in ("P0", "P1", "P2", "P3"))
            print(f"  {k:8} [{key[0]}/{key[1]}] {key[2][:40]:42} {cells}")
            for w in ("P0", "P1", "P2", "P3"):
                o = max(set(outs[key][w]), key=outs[key][w].count)
                print(f"             {w}: {o[:80]}")


if __name__ == "__main__":
    main()
