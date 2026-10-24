"""5B.12 on M3: the Session's `full_delay` (wait after the quick guess before the full search), typed as the user typed.
Prints numbers only (D-019).

  cd DIR && ~/Projects/llm-ime/.venv/bin/python scripts/delay_bench.py [--dir LOGDIR] [--n 120] [--progress FILE] DAY ...

DIR: this code (git archive) with models -> ~/Projects/llm-ime/models. Takes --n compositions the user committed with
space or a digit, typed without deleting (evenly spread), and replays each one's keys at the times they were pressed
into a Session as the engine service has it (k 5, more 20, segments), once for each setting of DELAYS, the order
rotating; at the commit's time it calls hurry() as the shell does when space waits for the full answer. Per key: the
time to its quick guess, and whether a full (or more) search was running when it came; per commit: how long space
would wait for the full answer (the shell gives up at 400 ms). It loads an engine beside the service, so it only
runs while the user is not typing: when the newest typing log changes it stops (exit 3), keeping what it has done in
--progress (only indices and times; deleted when it is done), and the next run goes on from there.
"""

import argparse
import collections
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typo_bench  # noqa: E402
import typo_stats  # noqa: E402
from ime import engine as engines  # noqa: E402
from ime.session import Session  # noqa: E402

DELAYS = {"0": 0.0, "40": 0.04, "70": 0.07}  # name -> full_delay s
PAUSE = 0.8  # s between compositions: "more" runs, as after a commit


def compositions(d, days, n):
    comps = []
    for day in days:
        p = os.path.join(d, day + ".jsonl")
        if not os.path.exists(p):
            continue
        for r in typo_stats.load(p):
            keys, ev = r["keys"], [e for e in r["events"] if len(e) == 2]
            if r["how"] not in ("space", "digit") or not keys.isalpha() or not keys.islower():
                continue
            if len(ev) != len(keys) or any(e[1] == "del" for e in ev):
                continue  # typed straight through, one event per key
            comps.append((r.get("context", ""), keys, [e[0] for e in ev], r["ms"]))
    return comps[:: max(1, len(comps) // n)][:n]


class Rec:
    def __init__(self):
        self.cond = threading.Condition()
        self.at = {}  # (keys, stage) -> first time it came

    def __call__(self, context, keys, cands, stage, **kw):
        with self.cond:
            self.at.setdefault((keys, stage), time.perf_counter())
            self.cond.notify_all()


def stoppable(eng):
    """Wraps eng.convert to note (start, end) of each search that can be stopped: the full and more ones."""
    spans, f = [], eng.convert

    def convert(*a, **kw):
        t = time.perf_counter()
        try:
            return f(*a, **kw)
        finally:
            if kw.get("stop"):
                spans.append((t, time.perf_counter()))

    eng.convert = convert
    return spans


def replay(ses, rec, spans, ctx, keys, times, commit_ms):
    with rec.cond:
        rec.at.clear()
    del spans[:]
    t0 = time.perf_counter()
    sent = {}
    for j, ms in enumerate(times, 1):
        time.sleep(max(0.0, t0 + ms / 1000 - time.perf_counter()))
        sent[j] = time.perf_counter()
        ses.type(ctx, keys[:j])
    time.sleep(max(0.0, t0 + commit_ms / 1000 - time.perf_counter()))
    commit = time.perf_counter()
    ses.hurry()
    with rec.cond:
        rec.cond.wait_for(lambda: (keys, "full") in rec.at, timeout=3)
    at, busy = dict(rec.at), list(spans)
    rows = []
    for j in sent:
        q = at.get((keys[:j], "quick"))
        rows.append(dict(quick=None if q is None else (q - sent[j]) * 1000,
                         running=any(t0 <= sent[j] < t1 for t0, t1 in busy)))
    full = at.get((keys, "full"))
    end = dict(full=None if full is None else (full - sent[len(keys)]) * 1000,
               space_wait=None if full is None else max(0.0, (full - commit) * 1000))
    return rows, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=typo_stats.DEFAULT_DIR)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--progress", default="/tmp/llmime-5b12/progress.jsonl")
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    start = typo_bench.newest_log(a.dir)
    comps = compositions(a.dir, a.days, a.n)
    done = {}
    if os.path.exists(a.progress):
        for line in open(a.progress, encoding="utf-8"):
            r = json.loads(line)
            done[r["i"]] = r["runs"]
    todo = [i for i in range(len(comps)) if i not in done]
    if todo:
        eng = engines.load("v02")
        eng.convert("", "nihao", k=1)
        rec, spans = Rec(), stoppable(eng)
        ses = Session(eng, rec, k=5, more=20, segments=True, two_pass=True)
        os.makedirs(os.path.dirname(a.progress), exist_ok=True)
        with open(a.progress, "a", encoding="utf-8") as f:
            for i in todo:
                if typo_bench.newest_log(a.dir) != start:
                    print(f"typing again: stopped, {len(done)}/{len(comps)} done")
                    ses.close()
                    sys.exit(3)
                ctx, keys, times, commit_ms = comps[i]
                names = list(DELAYS)
                names = names[i % len(names):] + names[: i % len(names)]
                runs = {}
                for name in names:
                    ses.full_delay = DELAYS[name]
                    ses._last = None
                    runs[name] = replay(ses, rec, spans, ctx, keys, times, commit_ms)
                    time.sleep(PAUSE)
                done[i] = runs
                f.write(json.dumps({"i": i, "runs": runs}) + "\n")
                f.flush()
        ses.close()
    report(done)
    os.remove(a.progress)


def report(done):
    keys = collections.defaultdict(list)  # (name, bucket) -> rows
    ends = collections.defaultdict(list)
    for runs in done.values():
        for name, (rows, end) in runs.items():
            for j, r in enumerate(rows, 1):
                keys[(name, typo_bench.bucket(j))].append(r)
            ends[name].append(end)
    pct = typo_bench.pct
    print(f"{len(done)} 个上屏的键串")
    for b in ("1-3", "4-8", "9+"):
        print(f"## {b} 键:quick p50 / p90 ms(键到时没在跑 / 在跑 full 或 more)、被下一键抢先没出 quick 的、在跑的占比")
        for name in DELAYS:
            rs = keys[(name, b)]
            if not rs:
                continue
            q = [r["quick"] for r in rs if r["quick"] is not None]
            qd = [r["quick"] for r in rs if r["quick"] is not None and not r["running"]]
            qr = [r["quick"] for r in rs if r["quick"] is not None and r["running"]]
            print(f"  delay {name:>3} ms: {len(rs)} 键 quick p50 {pct(q, .5):.0f} p90 {pct(q, .9):.0f}"
                  f"(没在跑 {pct(qd, .5):.0f} / 在跑 {pct(qr, .5):.0f}) 抢先 {sum(r['quick'] is None for r in rs) / len(rs):.1%}"
                  f" 在跑 {sum(r['running'] for r in rs) / len(rs):.1%}")
    print("## 上屏:最后一键到 full p50 / p90 ms,空格等 full p50 / p90 ms,等的占比")
    for name in DELAYS:
        es = ends[name]
        f = [e["full"] for e in es if e["full"] is not None]
        w = [e["space_wait"] for e in es if e["space_wait"] is not None]
        print(f"  delay {name:>3} ms: full p50 {pct(f, .5):.0f} p90 {pct(f, .9):.0f}; 空格等 p50 {pct(w, .5):.0f} "
              f"p90 {pct(w, .9):.0f},等了的 {sum(x > 0 for x in w) / max(1, len(w)):.1%};没 full 的 {len(es) - len(f)}")


if __name__ == "__main__":
    main()
