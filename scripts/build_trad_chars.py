"""Characters written only in traditional Chinese -> ime/trad_chars.txt (5B.7: 這樣 not on the first page).

  python scripts/build_trad_chars.py TSCharacters.txt STCharacters.txt

From OpenCC's character tables (https://github.com/BYVoid/OpenCC, data/dictionary, Apache-2.0): a character is
traditional only if OpenCC turns it into something else (這 -> 这) and it is not a simplified character itself
(乾 干, 於 于, 著 着 and 啰 stay). None of them is among LCCC's 3000 most common characters (ime/char_freq.tsv).
"""

import os
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ime", "trad_chars.txt")


def load(path):
    table = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        k, v = line.rstrip("\n").split("\t")
        table[k] = v.split()
    return table


def main():
    ts, st = load(sys.argv[1]), load(sys.argv[2])
    trad = sorted(c for c, v in ts.items() if len(c) == 1 and c not in v and c not in st)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# traditional-only characters, from OpenCC TSCharacters / STCharacters (Apache-2.0); "
                "scripts/build_trad_chars.py\n")
        f.write("".join(trad) + "\n")
    print(f"{len(trad)} characters -> {OUT}")


if __name__ == "__main__":
    main()
