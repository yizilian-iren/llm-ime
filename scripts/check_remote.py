"""Checks ime.remote (7E.1) with fake engines over a real TCP connection on 127.0.0.1, no model:

  .venv/bin/python scripts/check_remote.py

The remote answers when it is up (its token paths too); slower than quick_ms, the local engine answers and the remote
is left alone for backoff_s; a newer key (stop) drops the remote search; the service down, another model there, a
wrong token or the menu's 远程计算 off: the local engine, without waiting. Exits 1 when a check fails.
"""

import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import remote  # noqa: E402

remote.log = lambda msg: None


class Fake:
    def __init__(self, name, model="r6-4bit", delay=0.0):
        self.name, self.model_path, self.delay = name, f"models/{model}", delay
        self.paths, self.calls, self.stopped = {}, [], 0
        self.prefixes = {"ni", "hao"}

    def convert(self, context, keys, k=5, beam=None, mix=True, prefix=None, stop=None, eos=True):
        self.calls.append((keys, k, beam, prefix))
        end = time.monotonic() + self.delay
        while time.monotonic() < end:
            if stop and stop():
                self.stopped += 1
                return None
            time.sleep(0.002)
        out = [f"{self.name}:{keys}"]
        self.paths = {out[0]: ((7, len(keys)),)}
        return out

    def segments(self, context, keys):
        return [(self.name, 1)]

    def score(self, context, keys, texts):
        return [float(len(t)) for t in texts]


def start(engine, token="t0k"):
    svc = remote.Service(engine, token)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(4)

    def accept():
        while True:
            try:
                c, _ = s.accept()
            except OSError:
                return
            threading.Thread(target=svc.serve_conn, args=(c,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return s, s.getsockname()[1]


def connected(r, want=True, t=2.0):
    end = time.monotonic() + t
    while (r._sock is not None) != want and time.monotonic() < end:
        time.sleep(0.01)
    return (r._sock is not None) == want


fails = []


def check(what, ok):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


far = Fake("far")
srv, port = start(far)
local = Fake("near")
r = remote.RemoteEngine(local, "127.0.0.1", port, token="t0k", quick_ms=80, full_ms=300, backoff_s=0.5)
check("connects", connected(r))
out = r.convert("", "nihao", k=5)
check("the remote answers", out == ["far:nihao"] and r.last_by == "remote" and not local.calls)
check("its paths come back", r.paths == {"far:nihao": ((7, 5),)})
out = r.convert("", "nihao", k=3, beam=3, eos=False, prefix=((7, 2),))
check("quick options and the prefix go along", far.calls[-1] == ("nihao", 3, 3, [(7, 2)]))
check("segments and score from the remote", r.segments("", "ni") == [("far", 1)] and r.score("", "x", ["ab"]) == [2.0])
check("the rest is the local engine's", r.prefixes == {"ni", "hao"})

far.delay = 0.2  # slower than quick_ms
t = time.monotonic()
out = r.convert("", "hao", k=3, beam=3)
ms = (time.monotonic() - t) * 1000
check(f"slow remote: the local engine answers ({ms:.0f} ms)", out == ["near:hao"] and r.last_by == "local" and ms < 150)
t = time.monotonic()
r.convert("", "hao", k=5)
check("then local at once while backing off", r.last_by == "local" and (time.monotonic() - t) < 0.05)
time.sleep(0.25)
check("the remote dropped the search it was too slow for", far.stopped >= 1)
far.delay = 0.0
time.sleep(0.4)
check("after backoff_s the remote again", r.convert("", "hao") == ["far:hao"])

far.delay = 0.25
stop_at = time.monotonic() + 0.05
t = time.monotonic()
out = r.convert("", "nihaoma", k=5, stop=lambda: time.monotonic() > stop_at)
check("a newer key stops the remote search", out is None and (time.monotonic() - t) < 0.1)
time.sleep(0.1)
check("and the remote drops it", far.stopped >= 2)
far.delay = 0.0

r.set_on(False)
check("远程计算 off: local", r.convert("", "ni") == ["near:ni"])
r.set_on(True)
check("on again: remote", r.convert("", "ni") == ["far:ni"])

srv.close()
r._sock.shutdown(socket.SHUT_RDWR)  # the service going away
check("disconnect seen", connected(r, False))
t = time.monotonic()
check("down: local without waiting", r.convert("", "ni") == ["near:ni"] and (time.monotonic() - t) < 0.05)

srv2, port2 = start(Fake("other", model="personal-20261019-4bit"))
r2 = remote.RemoteEngine(Fake("near"), "127.0.0.1", port2, token="t0k")
check("another model there: not used", not connected(r2, True, t=0.5) and r2.convert("", "ni") == ["near:ni"])
srv3, port3 = start(Fake("far"), token="right")
r3 = remote.RemoteEngine(Fake("near"), "127.0.0.1", port3, token="wrong")
check("a wrong token is refused", not connected(r3, True, t=0.5) and r3.convert("", "ni") == ["near:ni"])
r4 = remote.RemoteEngine(Fake("near"), "127.0.0.1", 9, token="t0k", connect_s=0.2)  # nothing listens
t = time.monotonic()
check("nothing there: local without waiting", r4.convert("", "ni") == ["near:ni"] and (time.monotonic() - t) < 0.05)

sys.exit(1 if fails else 0)
