"""Answer times in the typing log by deployed version, split by whether the previous key's full answer had come
before the key (else the key's quick guess first waited for the forward of the full search then running; 5B.11).

  ssh m3 /usr/bin/python3 - [--dir DIR] [--side] DAY[,DAY...] NAME=[YYYY-MM-DDT]HH:MM ... < scripts/answer_wait.py

NAME=HH:MM: a version deployed at that local time (until the next one); with a date, from then on across the days
given. --side: the quick answers also by where they were computed (r remote, l local; 7E.1). Prints p50s and counts
only: the log stays on M3 (D-019). Standard library only.
"""

import argparse
import collections
import datetime
import json
import os


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else -1


def bucket(k):
    return "1-3" if k <= 3 else "4-8" if k <= 8 else "9+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/Library/Application Support/llm-ime/log"))
    ap.add_argument("--side", action="store_true", help="quick answers also by side (r / l)")
    ap.add_argument("day", help="a day, or several separated by commas")
    ap.add_argument("cuts", nargs="+")
    a = ap.parse_args()
    cuts = [c.split("=", 1) for c in a.cuts]
    stats = collections.defaultdict(list)
    lines = (line for day in a.day.split(",") for line in open(os.path.join(a.dir, day + ".jsonl"), encoding="utf-8"))
    for line in lines:
        r = json.loads(line)
        t = datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%dT%H:%M")
        v = None
        for name, start in cuts:
            if (t if "T" in start else t[11:]) >= start:
                v = name
        if not v:
            continue
        sent, n = {}, 0  # keys count -> ms it was last sent at
        for e in r["events"]:
            if len(e) != 2:
                continue
            n = n - 1 if e[1] == "del" else n + 1
            if n > 0:
                sent[n] = e[0]
        full_at = {k: sent[k] + ms for k, stage, ms in r["answers"] if stage == "full" and k in sent}
        by = r.get("by") or ["l"] * len(r["answers"])
        for (k, stage, ms), side in zip(r["answers"], by):
            if k not in sent:
                continue
            b = bucket(k)
            if stage == "full":
                stats[(v, b, "full")].append(ms)
            if stage != "quick":
                continue
            done = k == 1 or (k - 1 in full_at and full_at[k - 1] <= sent[k])
            stats[(v, b, "quick")].append(ms)
            stats[(v, b, "quick " + side)].append(ms)
            stats[(v, b, "quick, previous full done" if done else "quick, previous full running")].append(ms)
            if k - 1 in sent:
                stats[(v, b, "key gap")].append(sent[k] - sent[k - 1])
    kinds = ("quick", "quick, previous full done", "quick, previous full running", "full", "key gap")
    if a.side:
        kinds = ("quick", "quick r", "quick l") + kinds[1:]
    for b in ("1-3", "4-8", "9+"):
        print(f"## {b} keys (p50 ms, n)")
        for name, _ in cuts:
            print(name, " | ".join(f"{kd} {pct(stats[(name, b, kd)], .5)} ({len(stats[(name, b, kd)])})" for kd in kinds))


if __name__ == "__main__":
    main()
