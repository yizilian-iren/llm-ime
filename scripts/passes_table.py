"""7D.1 step 5: the one-pass trade-off, for the user's redesign: P@1 / P@5 and time per key for beam x k x mixed-in model.

  python scripts/passes_table.py acc [--items 200] [--forms full,initials,mixed]
  python scripts/passes_table.py time [--items 40] [--form full]

acc: the first --items eval items (all sources, as run_eval): the quick guess's way (a Session's `_quick`, typed one key
  at a time, whole search under 10 keys, from the last best for more), with each setting's beam / k / mix, on the last
  key; P@1 / P@5 per form. Candidates are the first k of the search, so P@5 of k=3 is P@3.
time: p50 of the 4-8 keys' quick guess per setting, the settings alternating key by key in one process.
Run it while the machine is quiet (uptime, sysctl vm.swapusage first).
"""

import argparse
import collections
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ime import engine as engines  # noqa: E402
from quick_split import pct, session  # noqa: E402
from run_eval import keys_of  # noqa: E402

SCHEDULES = [(5, 4, 3), (5, 4, 2), (6, 5, 4, 3, 2), (3, 4, 5), (3, 5, 3)]  # beam width per step; the last repeats (the last two
# are controls: widening, and wide in the middle)
SETTINGS = ([dict(beam=b, k=k, mix=m) for m, b, k in itertools.product((True, False), (3, 4, 5, 8, 12), (3, 5))]
            + [dict(beam=b, k=3, mix=True) for b in SCHEDULES])


def name(s):
    b = s["beam"] if isinstance(s["beam"], int) else "/".join(map(str, s["beam"]))
    return f"beam {b:>9} k {s['k']} {'mix' if s['mix'] else 'no mix'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["acc", "time"])
    ap.add_argument("--items", type=int, default=0)
    ap.add_argument("--forms", default="full,initials,mixed")
    ap.add_argument("--form", default="full")
    a = ap.parse_args()
    eng = engines.load_capped("v02")
    items = [json.loads(line) for line in open("data/private/eval.jsonl")]
    eng.convert("", "nihao", k=1)
    if a.mode == "acc":
        items = items[: a.items or 200]
        hits = collections.Counter()
        for r in items:
            for form in a.forms.split(","):
                keys = keys_of(r, form)
                if keys is None:
                    continue
                for st in SETTINGS:
                    s = session(eng, **st)
                    for j in range(1, len(keys) + 1):
                        c = s._quick(r["context"], keys[:j])
                    hits[name(st), form, "n"] += 1
                    hits[name(st), form, "p1"] += bool(c) and c[0] == r["target"]
                    hits[name(st), form, "p5"] += r["target"] in c[:5]
        print(f"## P@1 / P@5, first {len(items)} items")
        for st in SETTINGS:
            print(f"{name(st)}  " + "  ".join(
                f"{f}: {100 * hits[name(st), f, 'p1'] / hits[name(st), f, 'n']:.1f} / {100 * hits[name(st), f, 'p5'] / hits[name(st), f, 'n']:.1f}"
                for f in a.forms.split(",")))
    else:
        items = items[: a.items or 40]
        lat, n = collections.defaultdict(list), 0
        for r in items:
            keys = r[a.form]
            sess = [session(eng, **st) for st in SETTINGS]
            for j in range(1, len(keys) + 1):
                order = list(range(len(SETTINGS)))
                order = order[n % len(order):] + order[: n % len(order)]
                n += 1
                for i in order:
                    t0 = time.perf_counter()
                    sess[i]._quick(r["context"], keys[:j])
                    if 4 <= j <= 8:
                        lat[i].append((time.perf_counter() - t0) * 1000)
        print(f"## 4-8 keys quick p50 ms, {a.form}, {len(items)} items, load {os.getloadavg()[0]:.1f}")
        for i, st in enumerate(SETTINGS):
            print(f"{name(st)}  n={len(lat[i])}  p50 {pct(lat[i], .5):5.1f}  mean {sum(lat[i]) / len(lat[i]):5.1f}")


if __name__ == "__main__":
    main()
