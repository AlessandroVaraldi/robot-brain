#!/usr/bin/env python3
"""How much does a result depend on the wording of the prompt?

With a small model, changing one sentence of the prompt can fix one answer and
break another, and repeating a run with the same prompt does not reveal this:
a case that passes every repetition may still be fragile.  So the whole answer
suite runs under four wordings of the SAME prompt - same content, different
words and order - and a case counts as solid only if it holds under all of them.
Results are written to eval/suite_out/.

    python3 eval/robust.py --model qwen3:8b --tag q8
    python3 eval/robust.py --model qwen3:8b --body --tag q8_body   # body facts in the self block
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "john"))
import two_stage as T  # noqa: E402
import suite  # noqa: E402

WORDINGS_DIR = HERE / "wordings"


def wording(name):
    """Another wording of the reply prompt, from eval/wordings/<name>.txt."""
    text = (WORDINGS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


P0 = T.TEXT_PROMPT
JSON_LINE = 'Reply with ONLY one JSON object: {"text":"..."}'
assert P0.endswith(JSON_LINE)

# Same content as P0.  Paraphrased: every sentence said differently.
P1 = wording("p1")
# P0's own sentences, in another order.
_s = P0.split("\n")
P2 = "\n".join([
    "You are John, an experimental Physical AI living in a small humanoid robot in "
    "a university research lab.",
    _s[2], _s[4], _s[3],
    "You have a camera, one hand and a voice, and you cannot walk or pick things up.",
    _s[1], _s[5]])
# The same content as a list.
P3 = wording("p3")
WORDINGS = {"P0": P0, "P1": P1, "P2": P2, "P3": P3}
TEXT_MODELS = ("qwen3:8b", "qwen3:14b", "gemma3:12b", "qwen3.5:latest")


def unload_all():
    import urllib.request
    for m in TEXT_MODELS + (T.VISION_MODEL,):
        req = urllib.request.Request("http://127.0.0.1:11434/api/generate",
                                     data=json.dumps({"model": m, "keep_alive": 0}).encode())
        try:
            urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            pass
    time.sleep(3)


def latency(n=10):
    seen = {"writing": "WHAT IS MY NAME?", "engaged": True, "scene": suite.SCENE}
    rec, mem = suite.CTX["known"]
    ts = []
    for _ in range(n):
        t = time.time()
        T.answer(seen, dict(rec), mem, suite.SELF)
        ts.append(time.time() - t)
    return statistics.median(ts) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--body", action="store_true")
    ap.add_argument("--clock", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    unload_all()
    T.TEXT_MODEL, T.BODY_IN_SELF = a.model, a.body
    T.CLOCK_FOR_TIME, T.CHECK_REPLIES = a.clock, a.check
    out = {"model": a.model, "body": a.body, "clock": a.clock, "check": a.check,
           "wordings": {}}
    for w, prompt in WORDINGS.items():
        T.TEXT_PROMPT = prompt
        out["wordings"][w] = suite.run_answer(a.n, 1)
        if w == "P0":
            out["latency_ms"] = latency()
    path = HERE / "suite_out" / f"robust_{a.tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{a.tag}: {path.name}  latency {out['latency_ms']:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
