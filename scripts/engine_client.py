"""Type into the engine service (ime.server) like the shell does, one key at a time.

  .venv/bin/python scripts/engine_client.py jintianqunachifan [--context 下班以后，] [--sock PATH] [--check]

Sends the growing key string after each quick answer, prints every answer with its time, and fails unless
every key got a quick answer and the last key a full one, none of them empty. --check also loads the engine here and fails
unless the full answer's first page (5) equals engine.convert(context, keys) (what the repl shows).
"""

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime.server import DEFAULT_SOCK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys")
    ap.add_argument("--context", default="")
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="the service's --k, for --check")
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(a.sock)
    s.settimeout(a.timeout)
    f = s.makefile("rb")

    def send(obj):
        s.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode())

    def recv():
        line = f.readline()
        if not line:
            sys.exit("service closed the connection")
        return json.loads(line)

    send({"op": "ping", "id": 0})
    assert recv() == {"id": 0, "op": "pong"}

    full = None
    for i in range(1, len(a.keys) + 1):
        keys = a.keys[:i]
        t = time.perf_counter()
        send({"id": i, "context": a.context, "keys": keys})
        want = ["quick", "full"] if i == len(a.keys) else ["quick"]
        for stage in want:
            r = recv()
            if r.get("id") != i or r.get("stage") != stage:
                sys.exit(f"key {i}: expected {stage}, got {r}")
            if not r["candidates"]:
                sys.exit(f"key {i}: {stage} answer is empty")
            print(f"{keys:24} {stage:5} {(time.perf_counter() - t) * 1000:6.0f} ms  {len(r['candidates']):2}  {' '.join(r['candidates'][:5])}")
            if stage == "full":
                full = r["candidates"]
    send({"op": "commit", "text": full[0]})
    s.close()

    if a.check:
        from ime import engine as engines
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        direct = engines.load("v02").convert(a.context, a.keys, k=a.k)
        # The first page must agree; further down, near-tied candidates can swap (batched GPU sums differ).
        if direct[:5] != full[:5]:
            sys.exit(f"service {full[:5]} != direct {direct[:5]}")
        same = sum(x == y for x, y in zip(full, direct))
        print(f"first page matches engine.convert; {same} of {len(direct)} in the same place")


if __name__ == "__main__":
    main()
