"""Chinese words -> English, for the English mode's pinyin (6B.2: pingguo -> apple) -> ime/zh_en.tsv.

  python scripts/build_zh_en.py path/to/cedict_1_0_ts_utf-8_mdbg.txt

CC-CEDICT (https://www.mdbg.net/chinese/dictionary?page=cc-cedict, CC BY-SA 4.0): each simplified word of up to
four characters with up to three short English renderings, taken from its senses in order (common nouns before
names, the reading with the most senses first): the phrases of a sense up to a comma, "to " and bracketed notes
dropped, at most three words. Senses that only point elsewhere (variant of, used in, see, surname, classifier,
abbr.) give none. The output is an adaptation of CC-CEDICT and is under CC BY-SA 4.0 too: its header says so and
where it comes from.
"""

import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ime", "zh_en.tsv")
LINE = re.compile(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/\s*$")
HAN = re.compile(r"^[一-鿿]{1,4}$")
SKIP = re.compile(r"^(?:(?:old |archaic |erhua |Japanese |Korean |Taiwan )?variant of|used in|see |surname|CL:|"
                  r"abbr\.|abbreviation|also written|also pr\.|classifier|(?:onom|interj)\.|unofficial )", re.I)
BRACKET = re.compile(r"\([^)]*\)|\[[^\]]*\]|\"")
WORDS = re.compile(r"^[A-Za-z][A-Za-z'\-]*(?: [A-Za-z][A-Za-z'\-]*){0,2}$")


def renderings(senses):
    out = []
    for s in senses:
        if SKIP.match(s.strip()):
            continue
        for part in s.split(";"):
            r = BRACKET.sub("", part.split(",")[0]).strip()
            r = " ".join((r[3:] if r.startswith("to ") else r).split())
            if WORDS.match(r) and r not in out:
                out.append(r)
            if len(out) == 3:
                return out
    return out


def main():
    src = sys.argv[1]
    header, words = [], {}
    for line in open(src, encoding="utf-8"):
        if line.startswith("#!"):
            header.append(line[2:].strip())
            continue
        m = LINE.match(line)
        if not m or not HAN.match(m.group(2)):
            continue
        words.setdefault(m.group(2), []).append((m.group(3), m.group(4).split("/")))
    date = next((h.split("=", 1)[1] for h in header if h.startswith("date=")), "?")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"# Chinese words -> English (6B.2), scripts/build_zh_en.py from CC-CEDICT {date}\n"
                "# CC-CEDICT: community maintained, published by MDBG, https://www.mdbg.net/chinese/dictionary?page=cc-cedict\n"
                "# (from CEDICT, Copyright (C) 1997, 1998 Paul Andrew Denisowski). This file is an adaptation of it\n"
                "# (short renderings only) and is licensed, like CC-CEDICT, under the Creative Commons\n"
                "# Attribution-ShareAlike 4.0 International License: https://creativecommons.org/licenses/by-sa/4.0/\n")
        n = 0
        for w in sorted(words):
            got = []
            # common nouns before names (苹果: apple, then Apple), the reading with the most senses first (打: hit)
            for _, senses in sorted(words[w], key=lambda e: (e[0][:1].isupper(), -len(e[1]))):
                got += [r for r in renderings(senses) if r not in got]
            if got:
                f.write(w + "\t" + "|".join(got[:3]) + "\n")
                n += 1
    print(f"{n} words -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
