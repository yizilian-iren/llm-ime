"""Where the user's commits start and end, against jieba's words (6B.5). Run on M3, where the typing log is (never copy
it off, D-019): prints counts only, and with --show the first-candidate misses of the kind below.

  ssh m3 'cd ~/Projects/llm-ime && .venv/bin/python - --show 10' < scripts/commit_bounds.py

start: of the commits of Han characters after a context ending in one, how many start where jieba (on the context's
last 16 characters + the text) puts a word boundary; against that, how many of the inner positions of the same texts
are boundaries (what gen_train.py's random cut points hit). end: the same for the end, where the next commit continues
this one (its context ends with this text). cross: the commits whose engine first candidate was not what was committed
and whose first candidate, unlike the commit, jieba joins into one word with the context's last character (还有只修 +
成与: 修成). split: the misses where the commit is one jieba word of two characters or more and the first candidate,
of as many characters, is not one (访客 committed over 房可, 弥补 over 密不).
"""

import argparse
import glob
import json
import os
import re

import jieba

HAN = re.compile(r"^[一-鿿]+$")
LOG = os.path.expanduser("~/Library/Application Support/llm-ime/log")


def cuts(text):
    at, out = 0, set()
    for w in jieba.lcut(text):
        at += len(w)
        out.add(at)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    jieba.setLogLevel(60)
    rows = []
    for p in sorted(glob.glob(os.path.join(LOG, "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("how") in ("space", "digit", "enter") and HAN.match(r.get("text") or ""):
                rows.append(r)
    start = [0, 0]
    inner = [0, 0]
    end = [0, 0]
    cross, missed, shown, split, split_shown = 0, 0, 0, 0, 0
    by_stage, cross_stage = {}, {}  # stage of the candidates shown at the commit -> split / cross misses
    for i, r in enumerate(rows):
        ctx, text = r.get("context") or "", r["text"]
        tail = re.search(r"[一-鿿]{1,16}$", ctx)
        if tail:
            t = tail.group()
            c = cuts(t + text)
            start[0] += len(t) in c
            start[1] += 1
            inner[0] += sum(len(t) + j in c for j in range(1, len(text)))
            inner[1] += len(text) - 1
        first = (r.get("cands") or [""])[0]
        if first and first != text and HAN.match(first) and len(first) == len(text) >= 2:
            if jieba.lcut(text) == [text] and len(jieba.lcut(first)) > 1:
                split += 1
                by_stage[r.get("stage")] = by_stage.get(r.get("stage"), 0) + 1
                if split_shown < a.show:
                    split_shown += 1
                    print(f"  split: [{ctx[-8:]}] {r.get('keys')}: first {first}, committed {text} ({r.get('stage')})")
        if tail:
            if first and first != text and HAN.match(first):
                missed += 1
                if len(t) not in cuts(t + first) and len(t) in c:
                    cross += 1
                    cross_stage[r.get("stage")] = cross_stage.get(r.get("stage"), 0) + 1
                    if shown < a.show:
                        shown += 1
                        print(f"  cross: [{t[-8:]}] {r.get('keys')}: first {first}, committed {text}",
                              f"({r.get('stage')})")
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if nxt and (nxt.get("context") or "").endswith(text) and nxt.get("app") == r.get("app"):
            head = re.match(r"[一-鿿]{1,16}", nxt["text"])
            if head:
                end[0] += len(text) in cuts(text + head.group())
                end[1] += 1
    pct = lambda x: f"{x[0]}/{x[1]} = {x[0] / max(1, x[1]):.1%}"  # noqa: E731
    print(f"commits {len(rows)}")
    print(f"start at a jieba boundary: {pct(start)}; inner positions that are: {pct(inner)}")
    print(f"end at a jieba boundary (next commit continues it): {pct(end)}")
    print(f"first candidate not committed (after Han context): {missed}; of them cross (joined to the context, the "
          f"commit not): {cross}")
    allmiss = sum(1 for r in rows if (r.get("cands") or [""])[0] != r["text"])
    print(f"first candidate not committed (all): {allmiss}; of them split (commit one word, first not): {split}")
    for st in sorted({r.get("stage") for r in rows}, key=str):
        n = sum(r.get("stage") == st for r in rows)
        m = sum(r.get("stage") == st and (r.get("cands") or [""])[0] != r["text"] for r in rows)
        print(f"  stage {st}: commits {n}, first not committed {m}, split {by_stage.get(st, 0)}, "
              f"cross {cross_stage.get(st, 0)}")


if __name__ == "__main__":
    main()
