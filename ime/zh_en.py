"""English for Chinese words, for pinyin typed in the English mode (6B.2: pingguo -> apple).

ime/zh_en.tsv (scripts/build_zh_en.py, from CC-CEDICT, CC BY-SA 4.0): word<TAB>English[|English...].
The engine reads the keys as Chinese first, so the candidates come in its order, then each word's renderings in the
dictionary's.
"""

import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zh_en.tsv")
_table = None


def table():
    global _table
    if _table is None:
        _table = {}
        for line in open(PATH, encoding="utf-8"):
            if line.startswith("#"):
                continue
            w, en = line.rstrip("\n").split("\t")
            _table[w] = en.split("|")
    return _table


def english(chinese, k=5):
    """The English of the Chinese candidates `chinese` (best first), whole candidates only, at most `k`."""
    t, out = table(), []
    for c in chinese:
        for e in t.get(c, ()):
            if e not in out:
                out.append(e)
    return out[:k]
