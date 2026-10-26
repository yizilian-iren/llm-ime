"""Quick answers around the remote engine's timeouts (7E.2): per timeout in engine.log, the remote's answers in the 30 s
before and after, the local ones in its backoff window; then all quick answers by side and by the n-th local answer
after a timeout; then the minutes where the local p50 was 200 ms or more (with or without a timeout).

  ssh m3 /usr/bin/python3 - DAY [DAY ...] < scripts/remote_windows.py

Counts and ms only: the log stays on M3 (D-019). Standard library only. An answer's time: the key's own time in the
composition plus its ms. The first local answer after a timeout includes the quick_ms waited for the remote.
"""

import bisect
import collections
import datetime
import json
import os
import sys

LOG = os.path.expanduser("~/Library/Application Support/llm-ime/log")
ENGINE_LOG = os.path.expanduser("~/Library/Logs/llm-ime/engine.log")
BACKOFF_S = 21


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else -1


def line(name, xs):
    return f"{name:14s} n={len(xs):5d} p50={pct(xs, .5):5d} p90={pct(xs, .9):5d} max={max(xs or [-1]):5d}"


def timeouts(day):
    out = []
    for l in open(ENGINE_LOG, encoding="utf-8"):
        if l.startswith(day) and "remote slower" in l:
            out.append(datetime.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S").timestamp())
    return out


def answers(day):
    """(when, ms, side "r" / "l") of each quick answer, by time."""
    rows = []
    for raw in open(os.path.join(LOG, day + ".jsonl"), encoding="utf-8"):
        r = json.loads(raw)
        t = datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00")).timestamp()
        start = t - r["ms"] / 1000
        sent, n = {}, 0
        for e in r["events"]:
            if len(e) != 2:
                continue
            n = n - 1 if e[1] == "del" else n + 1
            if n > 0:
                sent[n] = e[0]
        by = r.get("by") or ["l"] * len(r["answers"])
        for (k, stage, ms), side in zip(r["answers"], by):
            if stage == "quick" and k in sent:
                rows.append((start + (sent[k] + ms) / 1000, ms, side))
    return sorted(rows)


def main():
    for day in sys.argv[1:]:
        tos, rows = timeouts(day), answers(day)
        print(f"== {day}: {len(tos)} timeouts")
        for t0 in tos:
            pre = [ms for at, ms, s in rows if t0 - 30 <= at < t0 and s == "r"]
            loc = [ms for at, ms, s in rows if t0 <= at < t0 + BACKOFF_S and s == "l"]
            post = [ms for at, ms, s in rows if t0 + BACKOFF_S <= at < t0 + 50 and s == "r"]
            print(datetime.datetime.fromtimestamp(t0).strftime("%H:%M:%S"),
                  f"remote before n={len(pre):3d} p50={pct(pre, .5):4d} | local n={len(loc):3d} p50={pct(loc, .5):4d}"
                  f" max={max(loc or [-1]):5d} first={loc[0] if loc else -1:5d} | remote after p50={pct(post, .5):4d}")
        side, nth = collections.defaultdict(list), collections.defaultdict(list)
        last, idx = None, 0
        for at, ms, s in rows:
            i = bisect.bisect_right(tos, at) - 1
            if s == "r":
                side["remote"].append(ms)
            elif i >= 0 and at - tos[i] <= BACKOFF_S:
                side["local backoff"].append(ms)
                idx = idx + 1 if last == i else 1
                last = i
                nth[min(idx, 9)].append(ms)
            else:
                side["local other"].append(ms)
        for name, xs in sorted(side.items()):
            print(line(name, xs))
        for n, xs in sorted(nth.items()):
            print(line(f"local #{n}{'+' if n == 9 else ''}", xs))
        mins = collections.defaultdict(lambda: collections.defaultdict(list))
        for at, ms, s in rows:
            mins[datetime.datetime.fromtimestamp(at).strftime("%H:%M")][s].append(ms)
        for m in sorted(mins):
            loc, rem = mins[m]["l"], mins[m]["r"]
            if len(loc) >= 3 and pct(loc, .5) >= 200:
                print(f"slow minute {m}: local n={len(loc)} p50={pct(loc, .5)} max={max(loc)}"
                      f" | remote n={len(rem)} p50={pct(rem, .5)}")


if __name__ == "__main__":
    main()
