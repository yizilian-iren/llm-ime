"""The weekly retrain's samples from the typing log (6A.2): runs on M3, one JSON line per sample to stdout.

  cd ~/Library/Application\\ Support/llm-ime/engine && PYTHONPATH=. ~/Projects/llm-ime/.venv/bin/python - [--dir DIR] < scripts/retrain_samples.py

Read by scripts/retrain.py on mini, which keeps them only for the run (D-019, D-038: the user's own machines only,
deleted after). A sample is a commit of a candidate (picked, not the letters) whose text is 1-16 Han characters
the keys as typed read (a start of the keys, 4A.10: the keys it used), not deleted right after:
{"day", "context", "keys", "text", "pieces"} (pieces: each character's keys). The last line is the counts:
{"counts": {...}}.
"""

import argparse
import collections
import glob
import json
import os
import re

from ime.ngram import char_readings_table
from ime.userwords import spans

HAN = re.compile(r"[一-鿿]{1,16}")
LOG_DIR = os.path.expanduser("~/Library/Application Support/llm-ime/log")
CTX_CHARS = 64  # ime.fmt.CTX_CHARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=LOG_DIR)
    a = ap.parse_args()
    readings = char_readings_table(extra=True)
    counts = collections.Counter()
    for p in sorted(glob.glob(os.path.join(a.dir, "*.jsonl"))):
        day = os.path.basename(p)[:10]
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                counts["bad line"] += 1
                continue
            counts["records"] += 1
            text, keys = r.get("text"), r.get("keys") or ""
            if text is None or r.get("picked") is None:
                continue
            counts["candidate commits"] += 1
            if r.get("deleted_after", 0) > 0:
                continue
            if r.get("used"):
                keys = keys[: r["used"]]
            if not HAN.fullmatch(text) or not (keys.isascii() and keys.isalpha() and keys.islower()):
                continue
            counts["Han, keys a-z"] += 1
            sp = spans(text, keys, readings)
            if sp is None:  # a typo or a reading the table lacks: the keys do not read the text
                continue
            counts["keys read it"] += 1
            context = r.get("context") or ""
            print(json.dumps({"day": day, "context": context[-CTX_CHARS:], "keys": keys, "text": text,
                              "pieces": [keys[s:e] for s, e in sp]}, ensure_ascii=False))
    print(json.dumps({"counts": counts}))


if __name__ == "__main__":
    main()
