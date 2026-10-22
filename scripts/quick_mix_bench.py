"""5B.13: what the quick guess loses (and saves) when the mixed-in model scores fewer of its tokens, key by key in
one process. Per setting: p50 time, how often its first candidate is the full search's first, and (last key of
each item) P@1 against the target, for the first --items eval items. Prints numbers only.

  python scripts/quick_mix_bench.py [--items 60] [--forms full,initials]
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

SETTINGS = {"mix all": {}, "mix 1 step": dict(mix_steps=1), "mix 2 steps": dict(mix_steps=2), "no mix": dict(mix=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--forms", default="full,initials")
    a = ap.parse_args()
    eng = engines.load("v02")
    items = [json.loads(line) for line in open("data/private/eval.jsonl")][: a.items]
    eng.convert("", "nihao", k=1)
    res = {}
    n = 0
    for form in a.forms.split(","):
        for r in items:
            ctx, keys = r["context"], r[form]
            sess = {name: session(eng, **kw) for name, kw in SETTINGS.items()}
            fs = session(eng)
            for j in range(1, len(keys) + 1):
                full = fs._convert(ctx, keys[:j], k=5)
                fs._remember(ctx, keys[:j], full)
                names = list(SETTINGS)
                names = names[n % len(names):] + names[: n % len(names)]
                n += 1
                for name in names:
                    t0 = time.perf_counter()
                    c = sess[name]._quick(ctx, keys[:j])
                    ms = (time.perf_counter() - t0) * 1000
                    x = res.setdefault((form, name, bucket(j)), dict(ms=[], agree=[], p1=[]))
                    x["ms"].append(ms)
                    x["agree"].append(bool(c) and bool(full) and c[0] == full[0])
                    if j == len(keys):
                        x["p1"].append(bool(c) and c[0] == r["target"])
    for form in a.forms.split(","):
        print(f"## {form}, load {os.getloadavg()[0]:.1f}")
        for b in ("1-3", "4-8", "9+"):
            for name in SETTINGS:
                x = res[(form, name, b)]
                p1 = f"  P@1 {100 * sum(x['p1']) / len(x['p1']):.1f} ({len(x['p1'])})" if x["p1"] else ""
                print(f"{b:4s} {name:12s} n={len(x['ms']):4d} p50 {pct(x['ms'], .5):5.1f}  quick=full {100 * sum(x['agree']) / len(x['agree']):.1f}%{p1}")


if __name__ == "__main__":
    main()
