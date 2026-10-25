"""Words the user could not type whole (6B.7: jigexian 及格线 went up as 及, then 格线), on M3 where the typing log is (never
copied off, D-019): counts only.

  # on M3, as word_replay.py
  ssh m3 'cd ~/Projects/llm-ime && nice .venv/bin/python /tmp/llmime-6b7/scripts/word_pieces.py --model models/r6-4bit'

Pieces: a commit of Han characters whose keys are longer than what it read, followed within --gap seconds by commits
whose keys are each the end of the one before (what the first left typed), the texts together one word of ime/words.tsv
whose whole pinyin is the keys of the first. The replay: the first commit's context and keys through the engine
with and without word_top; whether the word is first / in the top 5, and the items where it gets in or out.
Also, for word_replay's word misses, the ones that drop out of the top 5 with word_top: how, as numbers only.
It runs only while the user is away (word_replay.wait_idle).
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys

import jieba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import word_replay as wr  # noqa: E402

from ime import engine as engines  # noqa: E402
from ime import llm  # noqa: E402


def pieces(table, gap):
    """[(context, keys, word, n pieces)]."""
    rows = []
    for p in sorted(glob.glob(os.path.join(wr.LOG, "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("how") in ("space", "digit", "enter") and wr.HAN.match(r.get("text") or ""):
                rows.append(r)
    out, i = [], 0
    when = lambda r: datetime.datetime.fromisoformat(r["t"])  # noqa: E731
    while i < len(rows):
        first = rows[i]
        keys, text, j, rest = first["keys"], first["text"], i + 1, first["keys"]
        while j < len(rows) and rows[j]["keys"] and rest.endswith(rows[j]["keys"]) and rows[j]["keys"] != rest \
                and (when(rows[j]) - when(rows[j - 1])).total_seconds() <= gap:
            text, rest, j = text + rows[j]["text"], rows[j]["keys"], j + 1
        if j > i + 1 and any(w == text for w, _s in table.get(keys, ())):
            out.append((first.get("context") or "", keys, text, j - i))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/r6-4bit")
    ap.add_argument("--gap", type=float, default=90)
    ap.add_argument("--top", type=int, default=100000)
    a = ap.parse_args()
    jieba.setLogLevel(60)
    table = llm._words(a.top)
    freq = {w: int(f) for w, f in (l.split("\t")[0::2] for l in open(llm.WORDS, encoding="utf-8") if not l.startswith("#"))}
    found = pieces(table, a.gap)
    misses, _ok = wr.items(0)
    print(f"pieces: {len(found)} words typed in 2 or more commits ({sum(n == 2 for *_x, n in found)} in 2); "
          f"word misses: {len(misses)}", flush=True)
    res = {}
    for name, opt in (("base", {}), ("word_top", {"word_top": a.top})):
        wr.wait_idle()
        eng = engines.load_capped("v02", model=a.model, **opt)
        res[name] = ([wr.convert(eng, c, k) for c, k, _w, _n in found], [wr.convert(eng, c, k) for c, k, _t in misses])
        del eng
    for name, (got, _m) in res.items():
        print(f"{name:8} pieces: first {sum(bool(x) and x[0] == w for (_c, _k, w, _n), x in zip(found, got))}, "
              f"top 5 {sum(w in x for (_c, _k, w, _n), x in zip(found, got))}", flush=True)
    b, w_ = res["base"][0], res["word_top"][0]
    print(f"         pieces: top 5 gained {sum(f[2] in y and f[2] not in x for f, x, y in zip(found, b, w_))}, "
          f"lost {sum(f[2] in x and f[2] not in y for f, x, y in zip(found, b, w_))}", flush=True)
    lost = [(t, x, y) for (_c, _k, t), x, y in zip(misses, res["base"][1], res["word_top"][1]) if t in x and t not in y]
    print(f"word misses lost from the top 5 with word_top: {len(lost)}", flush=True)
    for t, x, y in lost:
        new = [c for c in y if c not in x]  # what came in, and where the commit stood
        print(f"         commit place {x.index(t) + 1} -> {y.index(t) + 1 if t in y else 'out'}; came in: {len(new)}, "
              f"its frequency {[freq.get(c) for c in new]} against the commit's {freq.get(t)}", flush=True)


if __name__ == "__main__":
    main()
