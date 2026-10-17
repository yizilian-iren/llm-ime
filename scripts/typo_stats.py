"""What the keys typed, deleted and typed again differ in (5B.8): the kinds of typing errors in the typing log.

  ssh m3 /usr/bin/python3 - [--dir DIR] DAY ... < scripts/typo_stats.py

Reads the shell's typing log (scripts/typing_summary.py) on M3 and prints counts and examples only: the log
stays on M3 (D-019). An episode is keys deleted and typed again: each run of deletes followed by letters in one
composition (the keys before the deletes against the keys it ended with), and a composition given up (Esc,
deleted to nothing) and typed again within 10 s (its keys against the next one's). The keys before are matched
to the prefix of the keys after that is nearest (edit distance with swaps; the longest on a tie), and the edits
between them are the error.
Standard library only.
"""

import argparse
import collections
import datetime
import json
import os

DEFAULT_DIR = os.path.expanduser("~/Library/Application Support/llm-ime/log")
RETYPE_S = 10
EXAMPLES = 6

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
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def when(r):
    return datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00"))


def key_events(r):
    return [e[1] for e in r.get("events", []) if len(e) == 2]


def episodes(records):
    """(before, after, where, done) for each time keys were deleted and typed again; done: the record that committed
    all of the keys after, else None."""
    out = []
    for i, r in enumerate(records):
        evs = key_events(r)
        s, runs = "", []          # runs: keys before each run of deletes, and whether letters followed
        prev_del = False
        for k in evs:
            if k == "del":
                if not prev_del:
                    runs.append([s, False])
                s = s[:-1]
                prev_del = True
            else:
                if prev_del:
                    runs[-1][1] = True
                s += k
                prev_del = False
        given_up = r["how"] in ("esc", "deleted")
        final = s if not given_up else None
        if given_up:
            # the keys it had before it was given up: before its last run of deletes, or as it was at Esc
            last = runs[-1][0] if r["how"] == "deleted" and runs and not runs[-1][1] else s
            if r["how"] == "deleted" and runs and not runs[-1][1]:
                runs = runs[:-1]
            final = last
            if i + 1 < len(records):
                nxt = records[i + 1]
                gap = (when(nxt) - when(r)).total_seconds() - nxt["ms"] / 1000
                if gap <= RETYPE_S and nxt.get("text") is not None and nxt.get("keys") and last:
                    out.append((last, nxt["keys"], "given up", nxt))
        done = r if not given_up and r.get("text") is not None and r.get("keys") == final else None
        for before, followed in runs:
            if followed and final:
                out.append((before, final, "deleted", done))
    return out


def align(a, b):
    """Edits turning a into b (Damerau): [(op, a_char, b_char, i_in_a)], op in same/ins/del/sub/swap.
    ins: a has a letter b lacks (an extra key); del: a lacks b's letter (a missed key)."""
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


def kind(op, a, b, ops, k):
    """A name for one edit."""
    name, x, y, i = op
    if name == "swap":
        return "颠倒两个键"
    if name == "sub":
        if adjacent(x, y):
            return "按成相邻键"
        return "按错键(不相邻)"
    # neighbours in the intended keys: the letters of b around where it happened
    j = sum(1 for o in ops[:k] if o[0] != "ins")      # position in b
    prev_b = b[j - 1] if j > 0 else ""
    next_b = b[j] if j < len(b) else ""
    if name == "ins":
        if x == prev_b or x == next_b:
            return "多按一次(重复键)"
        if adjacent(x, prev_b) or adjacent(x, next_b):
            return "多按相邻键"
        return "多按别的键"
    # del: a missed key y
    if y == "h" and prev_b in "zcs":
        return "漏 zh/ch/sh 的 h"
    if y == "g" and prev_b == "n":
        return "漏 ng 的 g"
    if y == prev_b:
        return "漏按重复键"
    if y in "aeiouv":
        return "漏元音"
    return "漏辅音"


KIND = {"多按别的键": "extra", "多按相邻键": "extra", "多按一次(重复键)": "extra", "漏元音": "miss", "漏辅音": "miss",
        "漏 zh/ch/sh 的 h": "miss", "漏 ng 的 g": "miss", "漏按重复键": "miss", "按成相邻键": "near", "颠倒两个键": "swap",
        "按错键(不相邻)": "other"}


def cases(records):
    """Keys typed with one error, as the user would have gone on typing them, and what was committed in the end:
    [(context, keys, text, kind)] (kind extra / miss / near / swap / other), one per delete-and-retype that differs by
    one edit from the keys committed (not at their first key). keys: the keys before the deletes + the rest of the
    keys committed; for the 5B.8 check, run on M3 (scripts/typo_eval.py)."""
    out = []
    for before, after, _where, done in episodes(records):
        if done is None or len(before) < 2:
            continue
        j, dist, ops = nearest_prefix(before, after)
        edits = [(k, o) for k, o in enumerate(ops) if o[0] != "same"]
        if dist != 1 or edits[0][1][3] == 0:
            continue
        k, o = edits[0]
        out.append((done.get("context") or "", before + after[j:], done["text"], KIND[kind(o, before, after, ops, k)]))
    return out


def end(op):
    """The key of the keys before at which the error shows: the wrong key, the second of a swap, the key after a miss."""
    return op[3] + 1 if op[0] == "swap" else op[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("days", nargs="+")
    args = ap.parse_args()
    records = []
    for day in args.days:
        p = os.path.join(args.dir, day + ".jsonl")
        if os.path.exists(p):
            records += load(p)
    commits = sum(1 for r in records if r.get("text") is not None)
    eps = episodes(records)
    print(f"{len(records)} 条记录，上屏 {commits}；删了重打 {len(eps)} 次"
          f"（组内 {sum(1 for e in eps if e[2] == 'deleted')}，放弃后重打 {sum(1 for e in eps if e[2] == 'given up')}）")

    groups = collections.Counter()
    ex = collections.defaultdict(list)
    kinds = collections.Counter()
    kinds_ex = collections.defaultdict(list)
    tail = collections.Counter()
    pairs = collections.Counter()
    later = collections.Counter()     # kinds of errors typed past before the deletes
    for before, after, where, _done in eps:
        j, dist, ops = nearest_prefix(before, after)
        edits = [(k, o) for k, o in enumerate(ops) if o[0] != "same"]
        if dist == 0:
            g = "一样(删了又打同样的键)"
        elif len(before) == 1:
            g = "只打了 1 个键就改"
        elif edits and edits[0][1][3] == 0:
            g = "第 1 个键就不同(多半换了词)"
        elif dist == 1:
            g = "差 1 处"
        elif dist == 2 and len(before) >= 4:
            g = "差 2 处"
        else:
            g = "差很多(换了打法 / 换了词)"
        groups[g] += 1
        if len(ex[g]) < EXAMPLES:
            ex[g].append(f"{before}→{after}")
        if g in ("差 1 处", "差 2 处"):
            for k, o in edits:
                name = kind(o, before, after, ops, k)
                pairs[(name, o[1] or "", o[2] or "")] += 1
                if len(before) - 1 - end(o) >= 1:
                    later[name] += 1
                kinds[name] += 1
                if len(kinds_ex[name]) < EXAMPLES:
                    kinds_ex[name].append(f"{before}→{after}")
            # how many keys were typed after the first error before the deletes
            tail[min(len(before) - 1 - end(edits[0][1]), 3)] += 1

    print("\n## 删掉的键和重打的键差在哪")
    for g, n in groups.most_common():
        print(f"- {g}: {n}（{n / len(eps):.0%}）  例: {', '.join(ex[g])}")
    small = groups["差 1 处"] + groups["差 2 处"]
    total_edits = sum(kinds.values())
    print(f"\n## 差 1–2 处的 {small} 次里的 {total_edits} 处错")
    for name, n in kinds.most_common():
        print(f"- {name}: {n}（{n / total_edits:.0%}），其中错后又打了键才删 {later[name]}"
              f"  例: {', '.join(kinds_ex[name])}")
    print("\n## 最常见的具体错（类 键→该打的键）")
    print(", ".join(f"{n} {a or '·'}→{b or '·'} ×{c}" for (n, a, b), c in pairs.most_common(30)))
    print("\n## 第一处错之后又打了几个键才删（差 1–2 处）")
    for t in sorted(tail):
        print(f"- {'3+' if t == 3 else t}: {tail[t]}（{tail[t] / small:.0%}）")


if __name__ == "__main__":
    main()
