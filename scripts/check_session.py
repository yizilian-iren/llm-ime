"""Checks the Session's timing (ime.session) against a fake engine that takes 10 ms per search:

  .venv/bin/python scripts/check_session.py

5B.12: the full search starts `full_delay` after the quick guess, and not at all for keys overtaken meanwhile;
`hurry` starts it at once, but not for a request that came after it. 6B.2: the English mode answers once, with the
English of whole pinyin only. Exits 1 when a check fails.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import zh_en  # noqa: E402
from ime.session import Session  # noqa: E402


class Engine:
    def __init__(self):
        self.calls = []

    def convert(self, context, keys, k=5, stop=None, **kw):
        self.calls.append((keys, k))
        time.sleep(0.01)
        return {"pingguo": ["苹果", "平果"]}.get(keys, [keys.upper()])

    def whole_pinyin(self, keys):
        return keys in ("pingguo", "hao")


def main():
    bad = 0

    def check(ok, what):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += not ok

    eng, events, t0 = Engine(), [], [0.0]
    english = {}

    def on_result(c, keys, cands, stage, **kw):
        events.append(((time.perf_counter() - t0[0]) * 1000, keys, stage))
        if stage == "english":
            english[keys] = cands

    ses = Session(eng, on_result, full_delay=0.05)

    def at(keys, stage):
        return next((t for t, k, s in events if (k, s) == (keys, stage)), None)

    def start():
        events.clear()
        t0[0] = time.perf_counter()

    start()
    ses.type("", "ab")
    time.sleep(0.03)
    ses.type("", "abc")
    time.sleep(0.2)
    check([(k, s) for _, k, s in events] == [("ab", "quick"), ("abc", "quick"), ("abc", "full")],
          f"a key within the delay: no full answer for the one before it {[(k, s) for _, k, s in events]}")
    check(("ab", 5) not in eng.calls, "nor its full search")
    gap = at("abc", "full") - at("abc", "quick")
    check(55 <= gap <= 90, f"the full answer {gap:.0f} ms after the quick guess (delay 50 + search 10)")

    start()
    ses.type("", "x")
    time.sleep(0.015)
    ses.hurry()
    time.sleep(0.1)
    check(at("x", "full") is not None and at("x", "full") < 45, f"hurry: the full answer at {at('x', 'full') or -1:.0f} ms")

    start()
    ses.hurry()
    ses.type("", "y")
    time.sleep(0.15)
    check(at("y", "full") is not None and at("y", "full") >= 55, f"hurry before the keys: not for them, {at('y', 'full') or -1:.0f} ms")

    start()
    ses.type("", "p", tag=1)
    time.sleep(0.003)
    ses.type("", "q", tag=2)
    time.sleep(0.2)
    check(at("p", "full") is not None and at("q", "full") is not None, "another typist's keys do not drop a full search")

    zh_en.table()  # as the service does at start
    start()
    ses.type("", "pingguo", english=True)
    ses.type("", "neccesary", tag=3, english=True)
    time.sleep(0.1)
    check([(k, s) for _, k, s in events] == [("pingguo", "english"), ("neccesary", "english")],
          f"English mode: one answer each {[(k, s) for _, k, s in events]}")
    check(english.get("pingguo", [""])[0] == "apple" and english.get("neccesary") == [],
          f"pingguo -> apple first, neccesary (not pinyin) none: {english}")
    ses.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
