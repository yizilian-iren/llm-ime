"""Try the engines by hand.

  make repl                       # v0.2's engine (r2 + base model)
  make repl ARGS="--engines v02,llm,ngram"

Type keys (e.g. hyhdddsgbbkxwjzywhf) and press Enter to see each engine's candidates. The first engine
runs in a typing session (ime.session): its quick guess is printed first, then the full search.
  1-5      commit that candidate of the first engine into the context
  /c TEXT  set the context (text left of the cursor); /c alone clears it
  /q       quit
"""

import argparse
import os
import queue
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime.session import Session  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="v02")
    a = ap.parse_args()
    engs = []
    for name in a.engines.split(","):
        t = time.time()
        engs.append(engines.load(name))
        print(f"loaded {name} in {time.time() - t:.1f}s")
    results = queue.Queue()
    ses = Session(engs[0], lambda c, k, cands, stage: results.put((time.perf_counter(), cands, stage)))
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
        t = time.perf_counter()
        ses.type(ctx, keys)
        stage = ""
        while stage != "full":
            done, cands, stage = results.get()
            print(f"  {engs[0].name + ' ' + stage:10} {(done - t) * 1000:6.0f} ms  " + show(cands))
        last = cands
        for e in engs[1:]:
            t = time.perf_counter()
            cands = e.convert(ctx, keys, k=5)
            print(f"  {e.name:10} {(time.perf_counter() - t) * 1000:6.0f} ms  " + show(cands))
    ses.close()


def show(cands):
    return "  ".join(f"{j + 1}.{c}" for j, c in enumerate(cands))


if __name__ == "__main__":
    main()
