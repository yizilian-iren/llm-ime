"""The eval set of Chinese with English words in it (6B.1): the user's own sentences, as typed in one go.

  .venv/bin/python scripts/build_mixed_eval.py [--n 300]

From the user's messages in Claude Code transcripts (scripts/extract_user_text.py's choice of them), runs of Han
characters and English words with nothing else between them but spaces (我用 python 写), at least 2 Han and one
English word of 2 letters or more. The target is the run without spaces next to the English, its English
lowercase (the input method gives the letters as typed); its context is the message before it, as in eval.jsonl.
Forms: full (each Han in full pinyin, each English word as itself), initials (Han by first letters), mixed
(pinyin.mixed_parts for the Han). `en` lists the English words, `known` whether each is in ime/en_words.txt.
Out: data/private/mixed_eval.jsonl (the user's text: stays in data/private).
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_user_text import user_messages  # noqa: E402
from ime import pinyin  # noqa: E402
from ime.llm import _english_words  # noqa: E402

RUN = re.compile(r"(?:[一-鿿]+|[A-Za-z]+| )+")
PART = re.compile(r"[一-鿿]+|[A-Za-z]+")
OUT = "data/private/mixed_eval.jsonl"


def items(msg):
    for m in RUN.finditer(msg):
        parts = PART.findall(m.group())
        han = sum(len(p) for p in parts if not p.isascii())
        en = [p for p in parts if p.isascii()]
        # an English word of 2+ letters, not a lone letter (A 和 B)
        if han < 2 or not en or any(len(w) < 2 for w in en) or len(en) > 3 or han > 24:
            continue
        # spaces only next to English: 用 python 写, not two Chinese phrases with a space between
        if re.search(r"[一-鿿] +[一-鿿]", m.group()):
            continue
        yield m.start(), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    known = _english_words()
    out, seen = [], set()
    for msg in user_messages():
        for start, parts in items(msg):
            target = "".join(p.lower() if p.isascii() else p for p in parts)
            if target in seen:
                continue
            seen.add(target)
            ctx = msg[:start].rstrip()[-64:]
            rng = random.Random(target)
            full, initials, mixed = [], [], []
            for p in parts:
                if p.isascii():
                    full.append(p.lower()); initials.append(p.lower()); mixed.append(p.lower())
                else:
                    s = pinyin.syllables(p)
                    full += s
                    initials.append(pinyin.initials(s))
                    mixed += pinyin.mixed_parts(s, rng)
            en = [p.lower() for p in parts if p.isascii()]
            out.append({"id": "m" + hashlib.sha1((ctx + target).encode()).hexdigest()[:10], "src": "user",
                        "context": ctx, "target": target, "en": en, "known": [w in known for w in en],
                        "full": "".join(full), "initials": "".join(initials), "mixed": "".join(mixed)})
    random.Random(0).shuffle(out)
    out = out[: a.n]
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    allk = sum(all(r["known"]) for r in out)
    print(f"{len(out)} items; all English words in en_words.txt: {allk}")


if __name__ == "__main__":
    main()
