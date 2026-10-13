"""How long typing waits on the engine service, typed like a person: one key every --gap ms, not waiting.

  python scripts/bench_keys.py [--sock PATH] [--gap 150] [--runs 1]

For each sentence's keys (full pinyin and initials, below) it sends the growing key string every --gap ms
and records, per key, the time from sending it to its first answer (the quick guess; a key overtaken by
the next one before any answer counts as skipped), and for the last key the time to the full answer.
Prints p50 / p90 of both, and how often the quick guess's first candidate already is the full answer's.
Talks to the service only through its socket, so it measures whatever service is running (its --k).
"""

import argparse
import json
import os
import socket
import statistics
import sys
import threading
import time

DEFAULT_SOCK = os.path.expanduser("~/Library/Application Support/llm-ime/engine.sock")  # as ime.server; no import,
# so the script runs on its own against a deployed service

SENTENCES = [  # (context, keys)
    ("", "nihao"), ("", "jintiantianqibucuo"), ("下班以后，", "jintianqunachifan"), ("", "wojuedezhegefangfakeyi"),
    ("", "mingtianjian"), ("", "xiexienidebangzhu"), ("", "zhegewentiwomenxiazhoutaolun"),
    ("我觉得", "haixingba"), ("", "dengyixiawozaikankan"), ("", "nizaiganshenme"),
    ("", "wmxzqcfm"), ("", "jtqndcf"), ("", "hyhdddsgbbkxwjzywhf"), ("", "zgsyqcyl"), ("", "wybzd"),
    ("", "dajiahao"), ("", "shiyanjieguohenhao"), ("", "womenkaihuiba"), ("好的，", "wozhidaole"), ("", "xxnlbz"),
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--gap", type=float, default=150, help="ms between keys")
    ap.add_argument("--runs", type=int, default=1)
    a = ap.parse_args()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(a.sock)
    f = s.makefile("rb")
    sent, first, full = {}, {}, {}  # id -> send time / first answer time / (full time, candidates)
    quick_top = {}
    done = threading.Event()

    def reader():
        for line in f:
            r = json.loads(line)
            t = time.perf_counter()
            i = r.get("id")
            if i is None:
                continue
            first.setdefault(i, t)
            if r["stage"] == "quick":
                quick_top.setdefault(i, r["candidates"][:1])
            if r["stage"] == "full":
                full[i] = (t, r["candidates"])
                if i in last_ids:
                    done.set()

    last_ids = set()
    threading.Thread(target=reader, daemon=True).start()
    rid = 0
    quick_ms, full_ms, skipped, agree, n_keys = [], [], 0, 0, 0
    for _ in range(a.runs):
        for context, keys in SENTENCES:
            ids = []
            for j in range(1, len(keys) + 1):
                rid += 1
                ids.append(rid)
                if j == len(keys):
                    last_ids.add(rid)
                    done.clear()
                sent[rid] = time.perf_counter()
                s.sendall((json.dumps({"id": rid, "context": context, "keys": keys[:j]}, ensure_ascii=False) + "\n").encode())
                if j < len(keys):
                    time.sleep(a.gap / 1000)
            if not done.wait(30):
                print(f"no full answer for {keys}", file=sys.stderr)
            time.sleep(0.3)  # a pause between sentences, like committing one
            last = ids[-1]
            for i in ids:
                n_keys += 1
                if i in first:
                    quick_ms.append((first[i] - sent[i]) * 1000)
                else:
                    skipped += 1
            if last in full:
                full_ms.append((full[last][0] - sent[last]) * 1000)
                agree += quick_top.get(last) == full[last][1][:1]
    s.close()
    print(f"{n_keys} keys, gap {a.gap:.0f} ms: first answer p50 {pct(quick_ms, 50):.0f} ms, p90 {pct(quick_ms, 90):.0f} ms, "
          f"{skipped} keys overtaken before any answer")
    print(f"{len(full_ms)} sentences: full answer after the last key p50 {pct(full_ms, 50):.0f} ms, "
          f"p90 {pct(full_ms, 90):.0f} ms, mean {statistics.mean(full_ms):.0f} ms; "
          f"quick's first = full's first on {agree} of {len(full_ms)}")


if __name__ == "__main__":
    main()
