"""Han character counts over LCCC (everyday conversation, simplified Chinese) -> ime/char_freq.tsv.

The engine ranks the characters of a single typed syllable by them (5B.5: wo -> 我 before 沃); traditional and
rare characters come out low. Characters seen fewer than MIN_COUNT times are left out (they get the lowest score).
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import corpus  # noqa: E402
from ime.ngram import CHAR_FREQ, char_readings_table  # noqa: E402

MIN_COUNT = 3


def main():
    chars = char_readings_table()
    c = Counter()
    for u in corpus.lccc():
        c.update(ch for ch in u if ch in chars)
    with open(CHAR_FREQ, "w", encoding="utf-8") as f:
        for ch, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            if n >= MIN_COUNT:
                f.write(f"{ch}\t{n}\n")
    print(f"{sum(c.values())} characters, {len(c)} distinct -> {CHAR_FREQ}")


if __name__ == "__main__":
    main()
