"""What a key cost the input method's main thread, and how much the candidates jumped, by app (6A.4).

  ssh m3 /usr/bin/python3 - [--dir DIR] [--from YYYY-MM-DDTHH:MM] [--min N] DAY ... < scripts/shell_time.py

Reads the typing log's `shell`, `busy`, `flips`, `ctx_ms` and `late_full` fields (shell/Sources/TypingLog.swift) of the
given days, from the deploy that wrote them (--from, local time). Prints counts and p50 / p90 / max only: the log
stays on M3 (D-019). Apps with fewer than --min keys (default 100) are counted together as "other". Standard library only.

  keys: ms from the key's own time to handle() (lag: the main thread was busy or the app slow to pass it on; keys replayed
        after a wait left out), ms in handle() (total), of it in setMarkedText, and in showing the candidates;
  ctx:  the first key's read of the text before the cursor from the app;
  busy: what showing an answer took the main thread, by stage;
  jumps per commit: times the first five / the first on screen changed, and how many of them were a full answer
        replacing a quick guess for the same keys (the others: a new key's answer over the old candidates);
  late: a quick guess committed before the full answer: how many full answers came (within 1.5 s) and had the same first.
"""

import argparse
import collections
import datetime
import json
import os


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else -1


def line(name, xs):
    return f"{name} p50 {pct(xs, .5):g} p90 {pct(xs, .9):g} max {max(xs) if xs else -1:g} (n={len(xs)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/Library/Application Support/llm-ime/log"))
    ap.add_argument("--from", dest="since", default="")
    ap.add_argument("--min", type=int, default=100)
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    recs = []
    for day in a.days:
        for l in open(os.path.join(a.dir, day + ".jsonl"), encoding="utf-8"):
            r = json.loads(l)
            t = datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%dT%H:%M")
            if "shell" in r and t >= a.since:
                recs.append(r)
    keys = collections.Counter()
    for r in recs:
        keys[r.get("app") or "?"] += len(r["shell"])
    big = {app for app, n in keys.items() if n >= a.min}
    groups = collections.defaultdict(list)
    for r in recs:
        app = r.get("app") or "?"
        groups[app if app in big else "other"].append(r)
    print(f"records {len(recs)} from {a.since or 'the start'}; keys by app: "
          + ", ".join(f"{k} {v}" for k, v in keys.most_common()))
    for app in sorted(groups, key=lambda g: -sum(len(r["shell"]) for r in groups[g])):
        rs = groups[app]
        print(f"## {app}: {len(rs)} records, {sum(len(r['shell']) for r in rs)} keys")
        sh = [e for r in rs for e in r["shell"]]
        print(" ", line("key lag ms", [e[1] for e in sh if e[1] >= 0]))
        print(" ", line("handle ms", [e[2] for e in sh]))
        print(" ", line("setMarkedText ms", [e[3] for e in sh]))
        print(" ", line("showing ms", [e[4] for e in sh if e[4] > 0]))
        print(" ", line("first key reads the text before the cursor ms", [r["ctx_ms"] for r in rs if "ctx_ms" in r]))
        for st in ("quick", "full", "more"):
            print(" ", line(f"answer shown, {st}, ms", [e[2] for r in rs for e in r.get("busy", []) if e[1] == st]))
        done = [r for r in rs if r.get("text") is not None]
        for name, pick in (("first five changed", lambda f: True), ("first changed", lambda f: f[4] == 1),
                           ("quick -> full, same keys, first changed", lambda f: f[3] == 1 and f[4] == 1)):
            dist = collections.Counter(min(sum(1 for f in r.get("flips", []) if pick(f)), 3) for r in done)
            print(f"  commits by times {name} (0 / 1 / 2 / 3+): "
                  + " / ".join(str(dist[i]) for i in range(4)) + f" of {len(done)}")
        lf = [r["late_full"] for r in done if "late_full" in r]
        came = [x for x in lf if x]
        print(f"  quick guesses committed before the full answer: {len(lf)}; the full answer came within 1.5 s {len(came)}"
              f", its first the one shown {sum(x[1] for x in came)}, the one committed {sum(x[2] for x in came)}"
              f", committed one in its first five {sum(x[3] for x in came)}; "
              + line("came after the commit ms", [x[0] for x in came]))


if __name__ == "__main__":
    main()
