"""First-candidate rate of engine v02 on the weekly retrain's test samples (6A.2), for each model given:

  .venv/bin/python scripts/retrain_eval.py SAMPLES.jsonl --days 2026-10-17,2026-10-18 --model models/r2-4bit \
      --model /path/to/new-4bit [--n 1500]

The same --n samples of those days (seeded) for every model, no user words. Prints one JSON line of rates and
counts, never a sample.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--days", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    days = set(a.days.split(","))
    items = []
    for line in open(a.samples, encoding="utf-8"):
        s = json.loads(line)
        if "counts" not in s and s["day"] in days:
            items.append(s)
    items = random.Random(a.seed).sample(items, min(a.n, len(items)))
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model paths are repo-relative
    out = {"n": len(items)}
    for m in a.model:
        eng = engines.load("v02", model=m)
        t, top1, top5 = time.time(), 0, 0
        for s in items:
            c = eng.convert(s["context"], s["keys"], k=5)
            top1 += bool(c) and c[0] == s["text"]
            top5 += s["text"] in c
        out[m] = {"p1": round(100 * top1 / len(items), 2), "p5": round(100 * top5 / len(items), 2),
                  "s": round(time.time() - t)}
        del eng
    print(json.dumps(out))


if __name__ == "__main__":
    main()
