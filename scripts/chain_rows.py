"""7D.3: the quick guess typed key by key from 10 keys on, going on from the last keys' paths: P@1 over the first --n public
items, per form, for the Session's `prefix_rows` (1: the last best only, as now), more rows, and `every` = the whole
search again each m-th key from 10 on. Also the quick search's time per key (M4 here: use onepass_bench on M3).

  python scripts/chain_rows.py [--forms full,initials,mixed] [--n 200] [--beam 12] [--rows 1,2,3,5] [--every 0,3]
"""

import argparse
import collections
import json
import os
import sys
import time
import resource

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ime import engine as engines  # noqa: E402
from quick_split import session  # noqa: E402
from run_eval import keys_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default="full,initials,mixed")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--rows", default="1,2,3,5")
    ap.add_argument("--every", default="0")
    a = ap.parse_args()
    mx.set_cache_limit(512 << 20)  # the run's many shapes would fill the buffer cache (17 GB at --n 100)
    eng = engines.load("v02")
    items = [json.loads(line) for line in open("data/private/eval.jsonl")]
    items = [r for r in items if r["src"] == "public"][: a.n]
    eng.convert("", "nihao", k=1)
    variants = [(r, e) for e in map(int, a.every.split(",")) for r in map(int, a.rows.split(","))]
    ok, ms = collections.defaultdict(list), collections.defaultdict(list)
    for form in a.forms.split(","):
        for r in items:
            keys = keys_of(r, form)
            if keys is None or len(keys) < 10:
                continue
            for rows, every in variants:
                s = session(eng, beam=a.beam)
                s.prefix_rows = rows
                for j in range(1, len(keys) + 1):
                    if every and j >= 10 and (j - 10) % every == 0:
                        s._last = None
                    t0 = time.perf_counter()
                    out = s._quick(r["context"], keys[:j])
                    if j >= 10:
                        ms[form, rows, every].append((time.perf_counter() - t0) * 1000)
                mx.clear_cache()
                ok[form, rows, every].append(bool(out) and out[0] == r["target"])
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.1f} GB")
    for form in a.forms.split(","):
        for rows, every in variants:
            o, m = ok[form, rows, every], sorted(ms[form, rows, every])
            if o:
                print(f"{form:8s} rows {rows} every {every}  n={len(o)}  P@1 {100 * sum(o) / len(o):.1f}  "
                      f"keys>=10 ms p50 {m[len(m) // 2]:.0f} p90 {m[int(len(m) * .9)]:.0f}")


if __name__ == "__main__":
    main()
