"""5B.13: quick guess and full search with the CPU work overlapped with the GPU (engine.overlap) against not, key by key
in one process, the order rotating; candidates compared. Prints numbers only.

  python scripts/overlap_bench.py [--items 40] [--form full]
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ime import engine as engines  # noqa: E402
from quick_split import bucket, pct, session  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--form", default="full", choices=["initials", "full", "mixed"])
    a = ap.parse_args()
    eng = engines.load("v02")
    items = [json.loads(line) for line in open("data/private/eval.jsonl")][: a.items]
    eng.convert("", "nihao", k=1)
    lat, same, n = {}, {"quick": [0, 0], "full": [0, 0]}, 0
    for r in items:
        ctx, keys = r["context"], r[a.form]
        sess = {o: session(eng) for o in (True, False)}
        for j in range(1, len(keys) + 1):
            order = (True, False) if n % 2 == 0 else (False, True)
            n += 1
            out = {}
            for o in order:
                eng.overlap = o
                s = sess[o]
                for stage in ("quick", "full"):
                    t0 = time.perf_counter()
                    c = s._quick(ctx, keys[:j]) if stage == "quick" else s._convert(ctx, keys[:j], k=5)
                    ms = (time.perf_counter() - t0) * 1000
                    if stage == "full":
                        s._remember(ctx, keys[:j], c)
                    out[(o, stage)] = c
                    lat.setdefault((o, stage, bucket(j)), []).append(ms)
            for stage in ("quick", "full"):
                same[stage][0] += out[(True, stage)] == out[(False, stage)]
                same[stage][1] += 1
                if out[(True, stage)] != out[(False, stage)]:
                    print('DIFF', stage, ctx[-8:], keys[:j], out[(True, stage)], out[(False, stage)])
    print(f"## overlap on / off, {a.form}, {n} keys, load {os.getloadavg()[0]:.1f}")
    for stage in ("quick", "full"):
        for b in ("1-3", "4-8", "9+"):
            on, off = lat[(True, stage, b)], lat[(False, stage, b)]
            print(f"{stage:5s} {b:4s} n={len(on):4d} on p50 {pct(on, .5):5.1f} off p50 {pct(off, .5):5.1f} "
                  f"({pct(on, .5) / pct(off, .5) - 1:+.0%})  mean {sum(on) / len(on):5.1f} / {sum(off) / len(off):5.1f} "
                  f"({sum(on) / sum(off) - 1:+.0%})")
        print(f"{stage}: same candidates {same[stage][0]}/{same[stage][1]}")


if __name__ == "__main__":
    main()
