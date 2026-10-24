"""7D.2 / D-053 on M3: where a key's quick answer spends its time with one pass, typed as the user typed. Prints
numbers only (D-019). Same replay as delay_bench.py (the user's key times into a Session as the engine service has it);
settings of the pause's "more" search: as deployed (more_delay 0.25 s), later (0.8 s), none.

  cd DIR && ~/Projects/llm-ime/.venv/bin/python scripts/onepass_bench.py [--dir LOGDIR] [--n 120] [--progress FILE] DAY ...

Per key: queue = key sent -> its quick search starts (the worker was busy: an older quick guess or a "more" search
running a beam step), compute = quick search start -> quick answer; and whether a "more" search was running when the
key came. Stops when the newest typing log changes (exit 3) and goes on from --progress next time, like delay_bench.
"""

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delay_bench as db  # noqa: E402
import typo_bench  # noqa: E402
import typo_stats  # noqa: E402
from ime import engine as engines  # noqa: E402
from ime.session import Session  # noqa: E402

SETTINGS = {"more 0.25": dict(more=20, more_delay=0.25, segments=True), "more 0.8": dict(more=20, more_delay=0.8, segments=True),
            "no more": dict(more=None, more_delay=0.25, segments=False)}
PAUSE = 1.2  # s between compositions (longer than the longest more_delay: "more" runs as after a commit)


def replay(ses, rec, spans, starts, ctx, keys, times):
    with rec.cond:
        rec.at.clear()
    del spans[:]
    starts.clear()
    t0 = time.perf_counter()
    sent = {}
    for j, ms in enumerate(times, 1):
        time.sleep(max(0.0, t0 + ms / 1000 - time.perf_counter()))
        sent[j] = time.perf_counter()
        ses.type(ctx, keys[:j])
    with rec.cond:
        rec.cond.wait_for(lambda: (keys, "quick") in rec.at, timeout=3)
    time.sleep(0.05)
    at, busy = dict(rec.at), list(spans)
    rows = []
    for j in sent:
        q, s = at.get((keys[:j], "quick")), starts.get(keys[:j])
        rows.append(dict(queue=None if s is None else (s - sent[j]) * 1000,
                         compute=None if q is None or s is None else (q - s) * 1000,
                         total=None if q is None else (q - sent[j]) * 1000,
                         more_running=any(t0 <= sent[j] < t1 for t0, t1 in busy)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=typo_stats.DEFAULT_DIR)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--progress", default="/tmp/llmime-7d2/progress.jsonl")
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    start = typo_bench.newest_log(a.dir)
    comps = db.compositions(a.dir, a.days, a.n)
    done = {}
    if os.path.exists(a.progress):
        for line in open(a.progress, encoding="utf-8"):
            r = json.loads(line)
            done[r["i"]] = r["runs"]
    todo = [i for i in range(len(comps)) if i not in done]
    if todo:
        eng = engines.load("v02")
        eng.convert("", "nihao", k=1)
        rec, spans = db.Rec(), db.stoppable(eng)
        ses = Session(eng, rec, k=5, more=20, segments=True)  # one pass, as deployed
        starts = {}
        quick = ses._quick

        def timed_quick(context, keys):
            starts[keys] = time.perf_counter()
            return quick(context, keys)

        ses._quick = timed_quick
        os.makedirs(os.path.dirname(a.progress), exist_ok=True)
        with open(a.progress, "a", encoding="utf-8") as f:
            for i in todo:
                if typo_bench.newest_log(a.dir) != start:
                    print(f"typing again: stopped, {len(done)}/{len(comps)} done")
                    ses.close()
                    sys.exit(3)
                ctx, keys, times, _commit = comps[i]
                names = list(SETTINGS)
                names = names[i % len(names):] + names[: i % len(names)]
                runs = {}
                for name in names:
                    for attr, v in SETTINGS[name].items():
                        setattr(ses, attr, v)
                    ses._last = None
                    runs[name] = replay(ses, rec, spans, starts, ctx, keys, times)
                    time.sleep(PAUSE)
                done[i] = runs
                f.write(json.dumps({"i": i, "runs": runs}) + "\n")
                f.flush()
        ses.close()
    report(done)
    os.remove(a.progress)


def report(done):
    rows = collections.defaultdict(list)
    for runs in done.values():
        for name, rs in runs.items():
            for j, r in enumerate(rs, 1):
                rows[(name, typo_bench.bucket(j))].append(r)
    pct = typo_bench.pct
    print(f"{len(done)} 个上屏的键串")
    for b in ("1-3", "4-8", "9+"):
        print(f"## {b} 键: total / queue / compute p50 (p90) ms;键到时 more 在跑的占比,在跑时 total p50 / 没在跑 total p50")
        for name in SETTINGS:
            rs = [r for r in rows[(name, b)] if r["total"] is not None]
            if not rs:
                continue

            def p(key, q, sel=rs):
                return pct([r[key] for r in sel if r[key] is not None], q)

            run = [r for r in rs if r["more_running"]]
            idle = [r for r in rs if not r["more_running"]]
            print(f"  {name:9s} {len(rs)} 键 total {p('total', .5):.0f} ({p('total', .9):.0f}) queue {p('queue', .5):.0f} ({p('queue', .9):.0f})"
                  f" compute {p('compute', .5):.0f} ({p('compute', .9):.0f}) 在跑 {len(run) / len(rs):.1%}"
                  f" total 在跑 {p('total', .5, run) if run else float('nan'):.0f} / 没在跑 {p('total', .5, idle):.0f}")


if __name__ == "__main__":
    main()
