"""5B.13: where the time of one key goes, quick guess and full search, in one process (M4; prints numbers only).

  python scripts/quick_split.py split [--items 40] [--form full]
  python scripts/quick_split.py paced [--items 20] [--gap 115]

split: on the first --items eval items, typed one key at a time as the Session does it (quick, then full k=5, the
  Session's own last best for the prefix): per key and stage the time in prefill (first model, mixed-in model), the
  readable-ness check, edges, the forwards (steps, rows), the softmax and copy to numpy, the end-of-search words
  (_whole_words, _single_first) and the rest (Python of the beam); then the quick guess without the mixed-in model
  and with a beam of 2, to see what each would save. Buckets by key count as the typing log's: 1-3, 4-8, 9+.
paced: the real Session thread, a key every --gap ms with jitter, as bench_keys does through the socket but without
  it: per key the wait (key sent -> quick starts, behind the previous key's full forward), the quick search itself,
  and the full search after the last key.
Run it while nothing else is (a busy machine slows everything alike).
"""

import argparse
import collections
import json
import os
import random
import sys
import threading
import time

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime.session import Session  # noqa: E402

PARTS = ("prefill", "mixprefill", "feasible", "edges", "softmax", "words", "forward", "python")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def bucket(n):
    return "1-3" if n <= 3 else "4-8" if n <= 8 else "9+"


class Timed:
    def __init__(self, eng):
        self.t = collections.Counter()
        self.eng = eng
        for obj, name, part in ((eng, "_prefill", "prefill"), (eng.mix, "prefill", "mixprefill"),
                                (eng, "feasible", "feasible"), (eng, "edges", "edges"),
                                (eng, "english_edges", "edges"), (eng, "idiom_edges", "edges"),
                                (eng, "_whole_words", "words"), (eng, "_single_first", "words")):
            self._wrap(obj, name, part)
        import ime.llm as L
        orig = L._log_softmax_cols

        def sm(logits, cols):  # the first step also pays here for the prefix forward (lazy), so it is not all softmax
            t0 = time.perf_counter()
            r = orig(logits, cols)
            mx.eval(r)
            self.t["softmax"] += (time.perf_counter() - t0) * 1000
            return r

        L._log_softmax_cols = sm

    def _wrap(self, obj, name, part):
        f = getattr(obj, name)

        def g(*a, **kw):
            t0 = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                self.t[part] += (time.perf_counter() - t0) * 1000

        setattr(obj, name, g)


def session(eng, **quick):
    s = Session.__new__(Session)
    s.engine, s.words, s.k = eng, None, 5
    s.quick, s.prefix_keys, s._last = {**dict(k=3, beam=3, mix=True, eos=False), **quick}, 10, None
    return s


def run_split(a):
    eng = engines.load("v02")
    timed = Timed(eng)
    items = [json.loads(line) for line in open(a.eval) if not a.src or json.loads(line)["src"] == a.src][: a.items]
    eng.convert("", "nihao", k=1)
    settings = {"quick": {}, "quick nomix": dict(mix=False), "quick beam2": dict(beam=2)}
    per = collections.defaultdict(list)
    for r in items:
        ctx, keys = r["context"], r[a.form]
        sess = {name: session(eng, **kw) for name, kw in settings.items()}
        for n in range(1, len(keys) + 1):
            for name, s in sess.items():
                runs = [("quick", lambda: s._quick(ctx, keys[:n]))]
                if name == "quick":
                    runs.append(("full", lambda: s._convert(ctx, keys[:n], k=5)))
                for stage, fn in runs:
                    timed.t.clear()
                    f0, c0 = eng.stats["forward_ms"], eng.stats["calls"]
                    t0 = time.perf_counter()
                    c = fn()
                    ms = (time.perf_counter() - t0) * 1000
                    if stage == "full":
                        s._remember(ctx, keys[:n], c)
                    x = dict(total=ms, forward=eng.stats["forward_ms"] - f0, steps=eng.stats["calls"] - c0,
                             **{p: timed.t[p] for p in PARTS if p != "forward"})
                    x["python"] = ms - sum(x[p] for p in PARTS if p != "python") - 0
                    label = stage if name == "quick" else name
                    per[(label, bucket(n))].append(x)
                    per[(label, "all")].append(x)
    print(f"## split, {a.form}, {len(items)} items, load {os.getloadavg()[0]:.1f}")
    for label in ("quick", "quick nomix", "quick beam2", "full"):
        for b in ("1-3", "4-8", "9+", "all"):
            xs = per.get((label, b))
            if not xs:
                continue
            tot = [x["total"] for x in xs]
            parts = "  ".join(f"{p} {sum(x[p] for x in xs) / len(xs):.1f}" for p in PARTS)
            print(f"{label:12s} {b:4s} n={len(xs):4d} total p50 {pct(tot, .5):5.1f} p90 {pct(tot, .9):5.1f} "
                  f"mean {sum(tot) / len(tot):5.1f} | steps {sum(x['steps'] for x in xs) / len(xs):.1f} | {parts}")
    print("(softmax includes the lazy prefix forward; python = total minus the others)")


def run_paced(a):
    eng = engines.load("v02")
    items = [json.loads(line) for line in open(a.eval) if not a.src or json.loads(line)["src"] == a.src][: a.items]
    eng.convert("", "nihao", k=1)
    rng = random.Random(1)
    log = {}  # id -> dict
    cond = threading.Condition()
    box = {}

    def on_result(ctx, keys, cands, stage, **kw):
        t = time.perf_counter()
        rec = log.get(len(keys) + 1000 * box["run"])
        if rec is not None:
            rec.setdefault(stage, t)
            if stage == "full":
                with cond:
                    cond.notify_all()

    s = Session(eng, on_result, k=5, two_pass=True)
    q0 = s._quick

    def timed_quick(ctx, keys):
        rec = log.get(len(keys) + 1000 * box["run"])
        if rec is not None:
            rec["quick_start"] = time.perf_counter()
        return q0(ctx, keys)

    s._quick = timed_quick
    rows = []
    for run, r in enumerate(items):
        box["run"] = run
        ctx, keys = r["context"], r[a.form]
        time.sleep(0.4)  # let the previous full finish
        for n in range(1, len(keys) + 1):
            rec = log.setdefault(n + 1000 * run, {})
            rec["sent"] = time.perf_counter()
            s.type(ctx, keys[:n])
            time.sleep(max(0.02, rng.gauss(a.gap, a.gap * 0.25) / 1000))
        rows.append((run, len(keys)))
    time.sleep(1)
    s.close()
    per = collections.defaultdict(list)
    for run, m in rows:
        for n in range(1, m + 1):
            rec = log[n + 1000 * run]
            if "quick" not in rec:
                per[bucket(n)].append(dict(skipped=1))
                continue
            x = dict(skipped=0, wait=(rec["quick_start"] - rec["sent"]) * 1000, search=(rec["quick"] - rec["quick_start"]) * 1000,
                     total=(rec["quick"] - rec["sent"]) * 1000)
            per[bucket(n)].append(x)
    print(f"## paced, gap {a.gap} ms, {len(items)} items, load {os.getloadavg()[0]:.1f}")
    for b in ("1-3", "4-8", "9+"):
        xs = [x for x in per[b] if not x["skipped"]]
        sk = sum(x["skipped"] for x in per[b])
        if not xs:
            continue
        waited = sum(x["wait"] > 5 for x in xs) / len(xs)
        print(f"{b:4s} n={len(xs):4d} skipped {sk}  quick total p50 {pct([x['total'] for x in xs], .5):5.1f} "
              f"p90 {pct([x['total'] for x in xs], .9):5.1f} | wait p50 {pct([x['wait'] for x in xs], .5):5.1f} "
              f"p90 {pct([x['wait'] for x in xs], .9):5.1f} (>5 ms on {waited:.0%}) | search p50 {pct([x['search'] for x in xs], .5):5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["split", "paced"])
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--form", default="full", choices=["initials", "full", "mixed"])
    ap.add_argument("--gap", type=float, default=115)
    ap.add_argument("--eval", default="data/private/eval.jsonl")
    ap.add_argument("--src", default="", help="only items from this source (user / public): home4070 gets public ones (7E.1)")
    a = ap.parse_args()
    (run_split if a.mode == "split" else run_paced)(a)


if __name__ == "__main__":
    main()
