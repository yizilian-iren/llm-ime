"""5B.11 on M3: what the typo readings (5B.8) cost per key, quick guess and full search, alternating key by key in one
process. Prints numbers only (D-019).

  cd DIR && ~/Projects/llm-ime/.venv/bin/python scripts/typo_bench.py [--dir LOGDIR] [--n 150] [--progress FILE] DAY ...

DIR: this code (git archive) with models -> ~/Projects/llm-ime/models. The keys are those of compositions the user
committed (--n of them, evenly spread), typed one key at a time as the Session does: quick (scripts' copy of
Session._quick, its own last best per setting), then full (k=5), for each setting of SETTINGS in turn, the order
rotating from key to key. Before each setting's quick the prompt is brought back to the previous key's and the
mixed-in model's context is already there, so each pays the same prefill as in typing. It loads an engine beside
the service, so it only runs while the user is not typing: when the newest typing log changes it stops (exit 3),
keeping what it has done in --progress (on M3, only indices and times; deleted when it is done), and the next run
goes on from there.
"""

import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typo_stats  # noqa: E402
from ime import engine as engines  # noqa: E402
from ime import fmt  # noqa: E402
from ime.session import Session  # noqa: E402

ON = dict(typo_penalty=4.0, swap_retries=2, typo_rows=3)  # 3538c52
SETTINGS = {  # engine attributes set for the full search; "quick": those set instead for the quick guess
    "off": {**ON, "typo_penalty": 0.0, "swap_retries": 1}, "on": ON, "rows2": {**ON, "typo_rows": 2},
    "rows1": {**ON, "typo_rows": 1}, "rows1 q0": {**ON, "typo_rows": 1, "quick": dict(typo_rows=0, swap_retries=1)}}
PARTS = ("edges", "feasible", "forward", "prefill", "steps", "rows")


def newest_log(d):
    return max(os.stat(p).st_mtime for p in glob.glob(os.path.join(d, "*.jsonl")))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


class Timed:
    """Wraps engine methods to add up the time spent in them per call of convert."""

    def __init__(self, eng):
        self.t = collections.Counter()
        for name in ("edges", "feasible", "_prefill"):
            self._wrap(eng, name, name.strip("_"))
        for name in ("_fork_cache", "_reorder"):  # rows of each forward, the mixed-in model's too
            self._count(eng, name)

    def _count(self, eng, name):
        f = getattr(eng, name)

        def g(cache_or_idx, *a):
            self.t["rows"] += len(a[0] if a and name == "_reorder" else cache_or_idx)
            return f(cache_or_idx, *a)

        setattr(eng, name, g)

    def _wrap(self, eng, name, part):
        f = getattr(eng, name)

        def g(*a, **kw):
            t0 = time.perf_counter()
            try:
                return f(*a, **kw)
            finally:
                self.t[part] += (time.perf_counter() - t0) * 1000

        setattr(eng, name, g)


def session(eng):
    s = Session.__new__(Session)  # its methods only, no worker thread
    s.engine, s.words, s.k = eng, None, 5
    s.quick, s.prefix_keys, s._last = dict(k=3, beam=3, mix=True, eos=False), 10, None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=typo_stats.DEFAULT_DIR)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--progress", default="/tmp/llmime-5b11/progress.jsonl")
    ap.add_argument("--overlap", action="store_true", help="the settings are engine.overlap off / on instead (5B.13)")
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    if a.overlap:
        SETTINGS.clear()
        SETTINGS.update({"off": {**ON, "overlap": False}, "on": {**ON, "overlap": True}})
    start = newest_log(a.dir)
    comps = []
    for day in a.days:
        p = os.path.join(a.dir, day + ".jsonl")
        if os.path.exists(p):
            comps += [(r.get("context", ""), r["keys"]) for r in typo_stats.load(p)
                      if r["how"] in ("space", "enter", "digit") and r["keys"].isalpha() and r["keys"].islower()]
    comps = comps[:: max(1, len(comps) // a.n)][: a.n]
    done = {}
    if os.path.exists(a.progress):
        for line in open(a.progress, encoding="utf-8"):
            r = json.loads(line)
            done[r["i"]] = r["keys"]
    todo = [i for i in range(len(comps)) if i not in done]
    if todo:
        eng = engines.load("v02")
        typos = eng.typos
        timed = Timed(eng)
        eng.convert("", "nihao", k=1)
        with open(a.progress, "a", encoding="utf-8") as f:
            for i in todo:
                if newest_log(a.dir) != start:
                    print(f"typing again: stopped, {len(done)}/{len(comps)} done")
                    sys.exit(3)
                ctx, keys = comps[i]
                sess = {name: session(eng) for name in SETTINGS}
                rows = []
                for n in range(1, len(keys) + 1):
                    names = list(SETTINGS)
                    names = names[(i + n) % len(names):] + names[: (i + n) % len(names)]
                    row, firsts = {}, {}
                    for name in names:
                        eng._prefill(fmt.prompt(ctx, keys[: n - 1], eng.order))
                        eng.mix.prefill(eng.tok.encode(ctx[-64:] if ctx.strip() else eng.empty_prompt))
                        s, row[name] = sess[name], {}
                        for stage in ("quick", "full"):
                            kw = {k: v for k, v in SETTINGS[name].items() if k != "quick"}
                            if stage == "quick":
                                kw.update(SETTINGS[name].get("quick", {}))
                            for k, v in kw.items():
                                setattr(eng, k, v)
                            eng.typos = typos if eng.typo_penalty else {}
                            timed.t.clear()
                            f0, r0, s0 = eng.stats["forward_ms"], eng.stats.get("swap_retries", 0), eng.stats["calls"]
                            t0 = time.perf_counter()
                            c = s._quick(ctx, keys[:n]) if stage == "quick" else s._convert(ctx, keys[:n], k=5)
                            if stage == "full":
                                s._remember(ctx, keys[:n], c)
                            row[name][stage] = dict(ms=(time.perf_counter() - t0) * 1000,
                                                    retries=eng.stats.get("swap_retries", 0) - r0,
                                                    forward=eng.stats["forward_ms"] - f0, steps=eng.stats["calls"] - s0,
                                                    **{p: timed.t[p] for p in ("edges", "feasible", "prefill", "rows")})
                            firsts[(name, stage)] = c[0] if c else ""
                    for name in names:
                        row[name]["quick"]["agree"] = firsts[(name, "quick")] == firsts[(name, "full")]
                        row[name]["full"]["same_as_off"] = firsts[(name, "full")] == firsts[("off", "full")]
                    rows.append(row)
                done[i] = rows
                f.write(json.dumps({"i": i, "keys": rows}) + "\n")
                f.flush()
    report(comps, done)
    os.remove(a.progress)


def bucket(n):
    return "1-3" if n <= 3 else "4-8" if n <= 8 else "9+"


def report(comps, done):
    per = collections.defaultdict(list)  # (setting, stage, bucket) -> [stats]
    for i, rows in done.items():
        for n, row in enumerate(rows, 1):
            for name, st in row.items():
                for stage, x in st.items():
                    per[(name, stage, bucket(n))].append(x)
                    per[(name, stage, "all")].append(x)
    print(f"{len(done)} 个上屏的键串，{sum(len(r) for r in done.values())} 键")
    for stage in ("quick", "full"):
        for b in ("1-3", "4-8", "9+", "all"):
            base = per[("off", stage, b)]
            if not base:
                continue
            p50o = pct([x["ms"] for x in base], 0.5)
            meano = sum(x["ms"] for x in base) / len(base)
            print(f"{stage} {b} 键 ({len(base)}):")
            for name in SETTINGS:
                xs = per[(name, stage, b)]
                ms = [x["ms"] for x in xs]
                p50, mean = pct(ms, 0.5), sum(ms) / len(ms)
                parts = "  ".join(f"{p} {sum(x[p] for x in xs) / len(xs):.1f}" for p in PARTS)
                extra = (f"  quick=full 第 1 {sum(x['agree'] for x in xs) / len(xs):.1%}" if stage == "quick" else
                         f"  第 1 同 off {sum(x['same_as_off'] for x in xs) / len(xs):.1%}")
                print(f"  {name:8s} p50 {p50:6.1f} ({p50 / p50o - 1:+.0%})  mean {mean:6.1f} ({mean / meano - 1:+.0%})"
                      f"  p90 {pct(ms, 0.9):6.1f}  重搜/键 {sum(x['retries'] for x in xs) / len(xs):.2f}  [{parts}]{extra}")


if __name__ == "__main__":
    main()
