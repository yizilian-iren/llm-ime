"""Try the engines by hand.

  make repl                       # v0.2's engine (r2 + base model)
  make repl ARGS="--engines v02,llm,ngram"

Type keys (e.g. hyhdddsgbbkxwjzywhf) and press Enter to see each engine's candidates.
  1-5      commit that candidate of the first engine into the context
  /c TEXT  set the context (text left of the cursor); /c alone clears it
  /q       quit
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="v02")
    a = ap.parse_args()
    engs = []
    for name in a.engines.split(","):
        t = time.time()
        engs.append(engines.load(name))
        print(f"loaded {name} in {time.time() - t:.1f}s")
    ctx, last = "", []
    while True:
        try:
            line = input(f"\n[{ctx[-30:]}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "/q":
            break
        if line.startswith("/c"):
            ctx = line[2:].strip()
            continue
        if line.isdigit() and last and 1 <= int(line) <= len(last):
            ctx += last[int(line) - 1]
            continue
        keys = "".join(ch for ch in line.lower() if ch.isalpha())
        if not keys:
            ctx += line  # punctuation etc. goes straight into the context
            continue
        for i, e in enumerate(engs):
            t = time.perf_counter()
            cands = e.convert(ctx, keys, k=5)
            ms = (time.perf_counter() - t) * 1000
            if i == 0:
                last = cands
            print(f"  {e.name:6} {ms:6.0f} ms  " + "  ".join(f"{j + 1}.{c}" for j, c in enumerate(cands)))


if __name__ == "__main__":
    main()
