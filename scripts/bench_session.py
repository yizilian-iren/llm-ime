"""Session latency (3A.2): quick guess vs full search, and simulated typing with stale work dropped.

  python scripts/bench_session.py [--engine v02] [--items 40] [--interval 150]

On the first --items eval items (--form, initials by default):
  step:   one key at a time, waiting for the full result each time: quick / full latency per key, and how
          often the quick guess's first candidate is the full search's first candidate.
  typing: keys arrive every --interval ms. Time from the last key to its quick and full results, for the Session
          (quick first, stale work dropped) and for converting every key in turn as the repl does (computed
          from the step run's full latencies).
"""

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime.session import Session  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


class Recorder:
    def __init__(self):
        self.events, self.cv = [], threading.Condition()

    def __call__(self, context, keys, cands, stage):
        with self.cv:
            self.events.append((time.perf_counter(), keys, cands, stage))
            self.cv.notify_all()

    def wait_full(self, keys, timeout=30):
        with self.cv:
            ok = self.cv.wait_for(lambda: any(e[1] == keys and e[3] == "full" for e in self.events), timeout)
            assert ok, f"no full result for {keys}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="v02")
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--interval", type=float, default=150)
    ap.add_argument("--form", default="initials", choices=["initials", "full", "mixed"], help="which keys of each item")
    ap.add_argument("--quick-beam", type=int, default=None, help="beam of the quick guess (Session's default if not given)")
    ap.add_argument("--opt", action="append", default=[], help="engine option key=value (numbers)")
    a = ap.parse_args()
    eng = engines.load(a.engine, **{k: float(v) for k, v in (o.split("=", 1) for o in a.opt)})
    items = [json.loads(line) for line in open("data/private/eval.jsonl")][: a.items]

    rec = Recorder()
    quick = None
    if a.quick_beam:
        quick = dict(k=3, beam=a.quick_beam, mix=True, eos=False)
    ses = Session(eng, rec, quick=quick)
    for r in items:
        r["initials"] = r[a.form]  # the rest of the script reads "initials"
    ses.type(items[0]["context"], items[0]["initials"])  # warm up
    rec.wait_full(items[0]["initials"])

    quick_ms, full_ms, agree, per_item = [], [], [], []
    for r in items:
        durs = []
        for j in range(1, len(r["initials"]) + 1):
            keys = r["initials"][:j]
            rec.events.clear()
            t0 = time.perf_counter()
            ses.type(r["context"], keys)
            rec.wait_full(keys)
            (tq, _, cq, _), (tf, _, cf, _) = rec.events[0], rec.events[-1]
            quick_ms.append((tq - t0) * 1000)
            full_ms.append((tf - tq) * 1000)
            durs.append(tf - tq)
            agree.append(bool(cq) and bool(cf) and cq[0] == cf[0])
        per_item.append(durs)
    print(f"## {a.engine} step ({len(quick_ms)} keys)")
    print(f"quick: p50 {pct(quick_ms, .5):.0f} p95 {pct(quick_ms, .95):.0f} ms; "
          f"full after it: p50 {pct(full_ms, .5):.0f} p95 {pct(full_ms, .95):.0f} ms; "
          f"quick top-1 = full top-1 on {100 * sum(agree) / len(agree):.1f}% of keys")

    gap = a.interval / 1000
    ses_lat, ses_quick, seq_lat = [], [], []
    for r, durs in zip(items, per_item):
        keys = r["initials"]
        rec.events.clear()
        t0 = time.perf_counter()
        for j in range(1, len(keys) + 1):
            while time.perf_counter() < t0 + (j - 1) * gap:
                time.sleep(0.001)
            ses.type(r["context"], keys[:j])
        t_last = t0 + (len(keys) - 1) * gap
        rec.wait_full(keys)
        tf = next(e[0] for e in rec.events if e[1] == keys and e[3] == "full")
        tq = next(e[0] for e in rec.events if e[1] == keys and e[3] == "quick")
        ses_lat.append((tf - t_last) * 1000)
        ses_quick.append((tq - t_last) * 1000)
        end = 0.0  # every key converted in turn: each starts when it arrives or when the last one ends
        for j, d in enumerate(durs):
            end = max(j * gap, end) + d
        seq_lat.append((end - (len(keys) - 1) * gap) * 1000)
        time.sleep(0.3)
    ses.close()
    print(f"## typing every {a.interval:.0f} ms ({len(items)} items): last key -> its full result")
    print(f"session: quick p50 {pct(ses_quick, .5):.0f} p95 {pct(ses_quick, .95):.0f} ms, "
          f"full p50 {pct(ses_lat, .5):.0f} p95 {pct(ses_lat, .95):.0f} ms; "
          f"every key in turn: p50 {pct(seq_lat, .5):.0f} p95 {pct(seq_lat, .95):.0f} ms")
    if "swap_retries" in eng.stats:
        print(f"second searches for a swap: {eng.stats['swap_retries']}")


if __name__ == "__main__":
    main()
