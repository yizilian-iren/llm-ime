"""Common words the engine can put into the beam whole (6B.7: jigexian 及格线) -> ime/words.tsv.

  .venv/bin/python scripts/build_words.py [--rows 100000]

The words of jieba's dictionary (https://github.com/fxsjy/jieba, MIT License, Copyright (c) 2013 Sun Junyi): Han only, 2 to 8
characters, not an idiom of ime/idioms.tsv, the most frequent first, with pypinyin's reading. Written as
`word<TAB>syllables<TAB>frequency`, tones dropped and ü as v, as the keys are typed.
"""

import argparse
import os
import re
import sys

import jieba
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_idioms import plain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAN = re.compile(r"^[一-鿿]{2,8}$")
HEADER = """# Words for 6B.7, scripts/build_words.py: the words of jieba's dictionary (https://github.com/fxsjy/jieba), MIT License,
# Copyright (c) 2013 Sun Junyi. Permission is hereby granted, free of charge, to any person obtaining a copy of these data,
# to deal in them without restriction, subject to this notice being included. The data is provided "as is", without
# warranty of any kind.
# word<TAB>syllables (tones dropped, ü as v)<TAB>frequency, most frequent first
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100000)
    a = ap.parse_args()
    idioms = {l.split("\t")[0] for l in open(os.path.join(ROOT, "ime", "idioms.tsv"), encoding="utf-8") if not l.startswith("#")}
    rows = []
    for line in open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8"):
        w, freq, _ = line.split()
        if HAN.match(w) and w not in idioms:
            rows.append((w, int(freq)))
    rows.sort(key=lambda r: (-r[1], r[0]))
    with open(os.path.join(ROOT, "ime", "words.tsv"), "w", encoding="utf-8") as f:
        f.write(HEADER)
        for w, freq in rows[: a.rows]:
            py = [plain(x) for x in lazy_pinyin(w, style=Style.NORMAL, v_to_u=False)]
            f.write(f"{w}\t{' '.join(py)}\t{freq}\n")
    print(f"{min(a.rows, len(rows))} words, the last with frequency {rows[min(a.rows, len(rows)) - 1][1]}")


if __name__ == "__main__":
    main()
