"""Idioms the engine gives a bonus when the keys read them whole (6B.3: zhigenzhidi 知根知底) -> ime/idioms.tsv.

  .venv/bin/python scripts/build_idioms.py path/to/chinese-xinhua/data/idiom.json

chinese-xinhua (https://github.com/pwxcoo/chinese-xinhua, MIT License, Copyright (c) 2018 PWXCOO): the idioms of 新华成语词典
with their pinyin. Kept: Han only, 4 to 12 characters, one syllable of pinyin per character. Also the words tagged `i`
in jieba's dictionary (MIT License, Copyright (c) 2013 Sun Junyi) used at least MIN_FREQ times, with pypinyin's reading:
知根知底 is in no dictionary of idioms. Written as `idiom<TAB>syllables`, tones dropped and ü as v, as the keys are typed.
"""

import json
import os
import re
import sys
import unicodedata

import jieba
from pypinyin import Style, lazy_pinyin

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ime", "idioms.tsv")
HAN = re.compile(r"^[一-鿿]+$")
MIN_FREQ = 10
HEADER = """# Idioms for 6B.3, scripts/build_idioms.py: the idioms of chinese-xinhua (https://github.com/pwxcoo/chinese-xinhua),
# MIT License, Copyright (c) 2018 PWXCOO, and the words tagged i in jieba's dictionary (https://github.com/fxsjy/jieba),
# MIT License, Copyright (c) 2013 Sun Junyi. Permission is hereby granted, free of charge, to any person obtaining a copy
# of these data, to deal in them without restriction, subject to this notice being included. The data is provided "as
# is", without warranty of any kind.
# idiom<TAB>syllables (tones dropped, ü as v)
"""


def plain(syllable):
    s = syllable.lower().replace("ü", "v")
    for a, b in zip("ǖǘǚǜ", "vvvv"):
        s = s.replace(a, b)
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def main():
    rows, seen = [], set()
    for r in json.load(open(sys.argv[1], encoding="utf-8")):
        w, py = r["word"].strip(), [plain(s) for s in r["pinyin"].split()]
        if not HAN.match(w) or not 4 <= len(w) <= 12 or len(py) != len(w) or not all(re.fullmatch("[a-z]+", s) for s in py):
            continue
        if (w, tuple(py)) not in seen:
            seen.add((w, tuple(py)))
            rows.append((w, py))
    have = {w for w, _ in rows}
    for line in open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8"):
        w, freq, tag = line.split()
        if tag == "i" and int(freq) >= MIN_FREQ and 4 <= len(w) <= 12 and HAN.match(w) and w not in have:
            rows.append((w, [plain(x) for x in lazy_pinyin(w, style=Style.NORMAL, v_to_u=False)]))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HEADER)
        for w, py in sorted(rows):
            f.write(f"{w}\t{' '.join(py)}\n")
    print(f"{len(rows)} idioms")


if __name__ == "__main__":
    main()
