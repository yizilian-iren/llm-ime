"""7B.2: per-key time of base models zero-shot (plain llm engine, no mix), alternating key by key in one process.

  ~/Projects/llm-ime/.venv/bin/python scripts/base_bench.py --models A,B [--n 40] [--beams 3,12] [--form mixed]

For each of the first --n public items (with context), the keys of --form are typed one at a time; at each key every
(model, beam) runs convert(k=5) in an order rotating from key to key, so a load change hits them alike. Keys 4-8 are
reported (the context is prefilled by then). Then the same keys again with overlap off, for the forward time of one
beam step (with overlap on, forward_ms is only the launch). Prefill: each item's context from an empty cache.
Run it while the machine is quiet; it prints numbers only.
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime.llm import LLMEngine  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--beams", default="3,12")
    ap.add_argument("--form", default="mixed")
    ap.add_argument("--eval", default="data/private/eval.jsonl")
    ap.add_argument("--keys", default="4-8")
    a = ap.parse_args()
    lo, hi = map(int, a.keys.split("-"))
    beams = [int(b) for b in a.beams.split(",")]
    models = a.models.split(",")
    engs = {m: LLMEngine(model=m) for m in models}
    items = [json.loads(line) for line in open(a.eval)]
    items = [r for r in items if r["src"] == "public" and r["context"] and len(r[a.form]) >= hi][: a.n]
    for e in engs.values():  # warm-up: trie, kernels
        for r in items[:3]:
            for b in beams:
                e.convert(r["context"], r[a.form][:hi], k=5, beam=b)
    runs = [(m, b) for m in models for b in beams]
    wall, steps, fwd = ({x: [] for x in runs} for _ in range(3))
    pre, toks = {m: [] for m in models}, {m: [] for m in models}
    turn = 0
    for r in items:
        for m, e in engs.items():
            e._ids, e._ctx_cache = [], None
            t0 = time.perf_counter()
            e._prefill(r["context"])
            pre[m].append((time.perf_counter() - t0) * 1000)
            toks[m].append(len(e.tok.encode(r["context"] + r["target"])) - len(e.tok.encode(r["context"])))
        for overlap in (True, False):
            for i in range(1, hi + 1):
                keys = r[a.form][:i]
                turn += 1
                order = runs[turn % len(runs):] + runs[: turn % len(runs)]
                for m, b in order:
                    e = engs[m]
                    e.overlap = overlap
                    s0 = dict(e.stats)
                    t0 = time.perf_counter()
                    e.convert(r["context"], keys, k=5, beam=b)
                    ms = (time.perf_counter() - t0) * 1000
                    n = e.stats["calls"] - s0["calls"]
                    if i < lo:
                        continue
                    if overlap:
                        wall[(m, b)].append(ms)
                        steps[(m, b)].append(n)
                    elif n:
                        fwd[(m, b)].append((e.stats["forward_ms"] - s0["forward_ms"]) / n)
    print(f"{len(items)} public items, form {a.form}, keys {lo}-{hi} ({len(wall[runs[0]])} calls per run)")
    print("model                     beam  p50 ms  p95 ms  steps p50  forward/step p50 ms  (overlap off)")
    for m, b in runs:
        print(f"{os.path.basename(m):26s}{b:4d}  {pct(wall[(m, b)], .5):6.1f}  {pct(wall[(m, b)], .95):6.1f}"
              f"  {statistics.median(steps[(m, b)]):9.1f}  {pct(fwd[(m, b)], .5):8.2f}")
    for m in models:
        print(f"{os.path.basename(m):26s}prefill p50 {pct(pre[m], .5):.1f} ms (context {statistics.median([len(r['context']) for r in items]):.0f} chars),"
              f" target tokens p50 {statistics.median(toks[m]):.0f}, mean {statistics.mean(toks[m]):.2f}")


if __name__ == "__main__":
    main()
