"""What the 打了又删、重打 of the daily summary is (6B.0): its compositions sorted by why the keys were typed again.

  ssh m3 /usr/bin/python3 - [--dir DIR] DAY ... < scripts/retype_stats.py

Reads the shell's typing log on M3 and prints counts only (D-019). A composition is in 打了又删 as in
scripts/typing_summary.py; its episodes (keys deleted and typed again, scripts/typo_stats.py) are sorted by the keys
before the deletes against the nearest prefix of the keys after:
  same      the same keys again: deleted and retyped as they were (the candidates for them were not the one wanted)
  typo      1 edit (2 for 4 keys or more), not at the first key: a typing error; 5B.8 reads an extra, missing or
            neighbouring key (typo:extra/miss/near), not a swap or other key (typo:swap / typo:other)
  first     1 edit, at the first key (of 2 keys or more)
  pieces    given up, then the start of its keys typed alone (as in 5A.5: typed again in pieces)
  fuller    the keys before are letters of the keys after, which are longer by 2 or more: initials or a shorter
            spelling that did not give the words, typed out (wmxz -> womenxianzai)
  short     only 1 key before the deletes
  rewrite   the rest: the first key differs or more than that apart: another word or another way of saying it
A composition with several episodes goes to the first kind in that order. Also: for typo:extra/miss/near, how many
keys were typed after the wrong one before deleting (0: the letter itself was seen, not the candidates); commits
deleted right after (deleted_after), those typed again within 10 s, and of those, the same keys again.
Standard library only.
"""

import argparse
import collections
import datetime
import os

DEFAULT_DIR = os.path.expanduser("~/Library/Application Support/llm-ime/log")
RETYPE_S = 10
ORDER = ["typo:extra", "typo:miss", "typo:near", "typo:swap", "typo:other", "first", "same", "pieces", "fuller",
         "short", "rewrite"]
ENGINE = {"typo:extra", "typo:miss", "typo:near", "same", "pieces", "fuller"}  # the input method could have spared


# --- from scripts/typo_stats.py and scripts/typing_summary.py (this runs on M3 as one file through stdin) ---
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
POS = {c: (y, x + (0, 0.25, 0.75)[y]) for y, row in enumerate(ROWS) for x, c in enumerate(row)}


def adjacent(a, b):
    if a not in POS or b not in POS or a == b:
        return False
    (ya, xa), (yb, xb) = POS[a], POS[b]
    if ya == yb:
        return abs(xa - xb) == 1
    return abs(ya - yb) == 1 and abs(xa - xb) <= 0.75


def load(path):
    import json
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def when(r):
    return datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00"))


def gap_to(r, nxt):
    return (when(nxt) - when(r)).total_seconds() - nxt["ms"] / 1000


def letters_then(events):
    seen_del = False
    for e in events:
        if e[1] == "del":
            seen_del = True
        elif seen_del and len(e) == 2 and len(e[1]) == 1:
            return True
    return False


def align(a, b):
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1] and a[i - 1] != a[i - 2]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d


def backtrack(d, a, b, j):
    i, ops = len(a), []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("same", a[i - 1], b[j - 1], i - 1)); i -= 1; j -= 1
        elif i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1] and d[i][j] == d[i - 2][j - 2] + 1:
            ops.append(("swap", a[i - 2:i], b[j - 2:j], i - 2)); i -= 2; j -= 2
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1], i - 1)); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("ins", a[i - 1], None, i - 1)); i -= 1
        else:
            ops.append(("del", None, b[j - 1], i)); j -= 1
    return ops[::-1]


def nearest_prefix(a, b):
    d = align(a, b)
    best = min(range(len(b) + 1), key=lambda j: (d[len(a)][j], -j))
    return best, d[len(a)][best], backtrack(d, a, b, best)
# --- end of the copies ---


def episodes(records, i):
    """(before, after, where) of record i: each run of deletes followed by letters (before: the keys then, after:
    the keys it ended with), and for one given up and typed again within 10 s, its keys against the next one's."""
    r = records[i]
    s, runs, prev_del = "", [], False
    for e in r.get("events", []):
        if len(e) != 2:
            continue
        if e[1] == "del":
            if not prev_del:
                runs.append([s, False])
            s, prev_del = s[:-1], True
        else:
            if prev_del:
                runs[-1][1] = True
            s, prev_del = s + e[1], False
    out = []
    final = s
    if r["how"] in ("esc", "deleted"):
        last = runs[-1][0] if r["how"] == "deleted" and runs and not runs[-1][1] else s
        if r["how"] == "deleted" and runs and not runs[-1][1]:
            runs = runs[:-1]
        final = last
        if i + 1 < len(records) and last:
            nxt = records[i + 1]
            if gap_to(r, nxt) <= RETYPE_S and nxt.get("keys"):
                out.append((last, nxt["keys"], "given up"))
    out += [(before, final, "deleted") for before, followed in runs if followed and final]
    return out


def subsequence(a, b):
    it = iter(b)
    return all(c in it for c in a)


def sort_episode(before, after, where):
    """(kind, keys typed after the error before the deletes: for a typo, else None)"""
    if where == "given up" and len(after) < len(before) and before.startswith(after):
        return "pieces", None
    j, dist, ops = nearest_prefix(before, after)
    edits = [o for o in ops if o[0] != "same"]
    if dist == 0:
        return "same", None
    if len(before) == 1:
        return "short", None
    if dist == 1 and edits[0][3] == 0:
        return "first", None
    if edits[0][3] == 0 or not (dist == 1 or (dist == 2 and len(before) >= 4)):
        if len(after) >= len(before) + 2 and subsequence(before, after):
            return "fuller", None
        return "rewrite", None
    o = edits[0]
    tail = len(before) - 1 - (o[3] + 1 if o[0] == "swap" else o[3])
    if o[0] == "swap":
        return "typo:swap", tail
    if o[0] == "sub":
        return ("typo:near" if adjacent(o[1], o[2]) else "typo:other"), tail
    return ("typo:extra" if o[0] == "ins" else "typo:miss"), tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    print("| 日 | 上屏 | 打了又删 | " + " | ".join(ORDER) + " | 没对上 | 输入法能省的 | 上屏后删 | 其中 10 s 内重打 |")
    print("|---" * (len(ORDER) + 7) + "|")
    tails, after_rows = collections.Counter(), []
    for day in a.days:
        p = os.path.join(a.dir, day + ".jsonl")
        if not os.path.exists(p):
            continue
        records = load(p)
        commits = sum(1 for r in records if r.get("text") is not None)
        kinds = collections.Counter()
        retype = 0
        for i, r in enumerate(records):
            retyped = letters_then(r.get("events", []))
            if r["how"] in ("esc", "deleted") and i + 1 < len(records):
                retyped = retyped or gap_to(r, records[i + 1]) <= RETYPE_S
            if not retyped:
                continue
            retype += 1
            got = dict(sort_episode(*e) for e in episodes(records, i))
            k = next((k for k in ORDER if k in got), "none")
            kinds[k] += 1
            if k in ("typo:extra", "typo:miss", "typo:near"):
                tails[(day, min(got[k], 3))] += 1
        after = [i for i, r in enumerate(records) if r.get("text") is not None and r.get("deleted_after", 0) > 0]
        again = [i for i in after if i + 1 < len(records) and gap_to(records[i], records[i + 1]) <= RETYPE_S]
        same = sum(1 for i in again if records[i + 1].get("keys") == records[i]["keys"])
        after_rows.append((day, len(again), same))
        engine = sum(kinds[k] for k in ENGINE)

        def share(n):
            return f"{n} ({100 * n / commits:.1f}%)" if commits else str(n)

        print(f"| {day} | {commits} | {share(retype)} | " + " | ".join(share(kinds[k]) for k in ORDER)
              + f" | {share(kinds['none'])} | {share(engine)} | {share(len(after))} | {share(len(again))} |")
    print("\n| 日 | typo:extra/miss/near:错键后又打 0 / 1 / 2 / 3+ 键才删 | 上屏后删、10 s 内重打 | 其中键串一样 |")
    print("|---|---|---|---|")
    for day, n, same in after_rows:
        print(f"| {day} | " + " / ".join(str(tails[(day, t)]) for t in range(4)) + f" | {n} | {same} |")


if __name__ == "__main__":
    main()
