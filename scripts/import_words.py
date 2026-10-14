"""Adds words to the running engine service's user word list (5A.3), e.g. ones taken from the user's own writing.

  python scripts/import_words.py FILE [--count 1] [--sock PATH]

FILE: `keys<TAB>word[<TAB>count]` lines (full or abbreviated pinyin, lowercase); count: how many picks a line without
one counts as. The service reads FILE itself (give a path on its Mac) and saves the list.
"""

import argparse
import json
import os
import socket

DEFAULT_SOCK = os.path.expanduser("~/Library/Application Support/llm-ime/engine.sock")  # as ime.server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    a = ap.parse_args()
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(30)
    s.connect(a.sock)
    s.sendall((json.dumps({"op": "import", "id": 1, "path": os.path.abspath(a.file), "count": a.count}) + "\n").encode())
    for line in s.makefile("rb"):
        r = json.loads(line)
        if r.get("id") == 1:
            print(f"imported {r['imported']}" if "imported" in r else r)
            break


if __name__ == "__main__":
    main()
