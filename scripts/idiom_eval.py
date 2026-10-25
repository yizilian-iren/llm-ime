"""Do common idioms and words still come out first? Two models side by side (the user's report, 10-19: idioms gone,
odd words first, after personal-20261018 and the Rime import).

  .venv/bin/python scripts/idiom_eval.py --model models/r2-4bit --model models/personal-20261018-4bit [--n 200]
  a model may carry engine options: --model models/r6-4bit@idiom_bonus=3,english_penalty=0.5

Public items only: the n most frequent idioms of jieba's dictionary (tag i, four characters) and n words of two or
three characters drawn at random from ranks 300-3000, typed as full pinyin and as initials, with no context, and
the extra items below. No user word list. Prints P@1 / P@5 per set and model, and the misses of the first model
that the other gets right or the other way round.
"""

import argparse
import os
import random
import re
import sys

import jieba
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402

HAN = re.compile(r"^[一-鿿]+$")
EXTRA = ["知根知底"]  # the user's report


def keys(text, initials):
    p = [s.replace("ü", "v") for s in lazy_pinyin(text, style=Style.NORMAL, v_to_u=False)]
    return "".join(s[0] for s in p) if initials else "".join(p)


def items(n):
    rows = [l.split() for l in open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8")]
    idi = sorted(((w, int(f)) for w, f, t in rows if t == "i" and len(w) == 4 and HAN.match(w)), key=lambda x: -x[1])
    idioms = [w for w, _ in idi[: n - len(EXTRA)]] + EXTRA
    words = sorted(((w, int(f)) for w, f, t in rows if 2 <= len(w) <= 3 and HAN.match(w)
                    and t in ("n", "v", "a", "d", "vn", "an", "ad", "l")), key=lambda x: -x[1])[300:3000]
    return {"idiom": idioms, "word": [w for w, _ in random.Random(7).sample(words, n)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    sets = items(a.n)
    res = {}
    for m in a.model:
        path, _, opt = m.partition("@")  # models/r6-4bit@idiom_bonus=3: engine options after @, comma separated
        eng = engines.load_capped("v02", model=path, **{k: float(v) for k, v in (o.split("=") for o in opt.split(",") if o)})
        for kind, ws in sets.items():
            for ini in (False, True):
                for w in ws:
                    c = eng.convert("", keys(w, ini), k=5)
                    res[(m, kind, ini, w)] = (c[0] if c else None, c.index(w) + 1 if w in c else None)
    for kind, ws in sets.items():
        for ini in (False, True):
            for m in a.model:
                r = [res[(m, kind, ini, w)] for w in ws]
                print(f"{kind:5} {'initials' if ini else 'full':8} {os.path.basename(m):34} n={len(ws)} "
                      f"P@1 {100 * sum(x[1] == 1 for x in r) / len(ws):.1f}  P@5 {100 * sum(x[1] is not None for x in r) / len(ws):.1f}")
    if len(a.model) == 2:
        m0, m1 = a.model
        for kind, ws in sets.items():
            for ini in (False, True):
                lost = [w for w in ws if res[(m0, kind, ini, w)][1] == 1 and res[(m1, kind, ini, w)][1] != 1]
                won = [w for w in ws if res[(m0, kind, ini, w)][1] != 1 and res[(m1, kind, ini, w)][1] == 1]
                print(f"{kind} {'initials' if ini else 'full'}: first -> second lost P@1 {len(lost)} {lost[:12]}; gained {len(won)} {won[:12]}")
    for ini in (False, True):
        for w in EXTRA:
            for m in a.model:
                print(f"{w} {'initials' if ini else 'full'} {os.path.basename(m)}: first {res[(m, 'idiom', ini, w)][0]}, rank {res[(m, 'idiom', ini, w)][1]}")


if __name__ == "__main__":
    main()
