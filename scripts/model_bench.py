"""6B.1: what a base model costs per key against another, alternating key by key in one process (a busy machine
slows both alike). Prints numbers only.

  python scripts/model_bench.py [--items 40] [--form full] [--a models/r2-4bit] [--b models/r6-4bit --b-opt english_penalty=0.5]

On the first --items eval items, typed one key at a time as run_eval's latency run does (full search, k=5): for each
key both engines convert the same prefix, the order rotating from key to key. Run it while nothing else is.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def opts(pairs):
    return {k: float(v) for k, v in (o.split("=", 1) for o in pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="v02")
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--form", default="full", choices=["initials", "full", "mixed"])
    ap.add_argument("--a", default="models/r2-4bit")
    ap.add_argument("--b", default="models/r6-4bit")
    ap.add_argument("--a-opt", action="append", default=[])
    ap.add_argument("--b-opt", action="append", default=["english_penalty=0.5"])
    ap.add_argument("--rounds", type=int, default=1, help="the items again, the order rotated by one")
    a = ap.parse_args()
    engs = [engines.load(a.engine, model=a.a, **opts(a.a_opt)), engines.load(a.engine, model=a.b, **opts(a.b_opt))]
    items = [json.loads(line) for line in open("data/private/eval.jsonl")][: a.items]
    for e in engs:  # warm up
        e.convert(items[0]["context"], items[0][a.form][:3], k=5)

    lat, n = ([], []), 0
    for rnd in range(a.rounds):
        for r in items:
            keys = r[a.form]
            for j in range(1, len(keys) + 1):
                order = (0, 1) if (n + rnd) % 2 == 0 else (1, 0)
                n += 1
                for i in order:
                    s = time.perf_counter()
                    engs[i].convert(r["context"], keys[:j], k=5)
                    lat[i].append((time.perf_counter() - s) * 1000)
    print(f"## {a.form}, {len(lat[0])} keys each, load {os.getloadavg()[0]:.1f}")
    for name, xs in zip((a.a, a.b), lat):
        print(f"{name}: p50 {pct(xs, .5):.1f} p90 {pct(xs, .9):.1f} p95 {pct(xs, .95):.1f} mean {sum(xs) / len(xs):.1f} ms")
    p50, p95 = pct(lat[1], .5) / pct(lat[0], .5), pct(lat[1], .95) / pct(lat[0], .95)
    print(f"b / a: p50 {100 * (p50 - 1):+.1f}%, p95 {100 * (p95 - 1):+.1f}%")


if __name__ == "__main__":
    main()
