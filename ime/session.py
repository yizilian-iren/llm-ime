"""Typing session: converts in a worker thread, quick guess first, full search after, stale work dropped.

Every keystroke calls `type(context, keys)`. The worker always takes the newest request (older ones still
queued are skipped) and for it:
  1. quick: a narrow beam starting from the best candidate it last had for a prefix of these keys (minus
     that candidate's last token, whose reading the new key may change), so only the tail is searched;
     with the mixed-in model too, since without it the quick guess agrees with the full search far less
     often (3A.2: 63% vs 40% of keys); top 3 only, and without the extra step that scores the end of the
     sentence (both cut its time by about a quarter for 1 point of agreement);
  2. full: the engine's normal search; dropped between beam steps as soon as a newer keystroke arrives.
`on_result(context, keys, candidates, stage)` is called from the worker thread with stage "quick" / "full".
All model work stays on the worker thread (MLX state is not shared between threads).
"""

import queue
import threading


class Session:
    def __init__(self, engine, on_result, k=5, quick=None):
        self.engine, self.on_result, self.k = engine, on_result, k
        self.quick = quick or dict(k=3, beam=3, mix=True, eos=False)  # convert() options of the quick guess
        self._q = queue.Queue()
        self._last = None  # (context, keys, best path) of the latest result, quick or full
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def type(self, context, keys):
        self._q.put((context, keys))

    def close(self):
        self._q.put(None)
        self._thread.join()

    def _newer(self):
        return not self._q.empty()

    def _run(self):
        while True:
            req = self._q.get()
            while req is not None and not self._q.empty():  # only the newest keystroke matters
                req = self._q.get()
            if req is None:
                return
            context, keys = req
            if not keys:
                self.on_result(context, keys, [], "full")
                continue
            quick = self._quick(context, keys)
            self.on_result(context, keys, quick, "quick")
            if self._newer():
                continue
            full = self.engine.convert(context, keys, k=self.k, stop=self._newer)
            if full is None:
                continue
            self._remember(context, keys, full)
            self.on_result(context, keys, full, "full")

    def _quick(self, context, keys):
        prefix = None
        if self._last and self._last[0] == context and keys.startswith(self._last[1]) and self._last[2]:
            prefix = list(self._last[2][:-1]) or None
        cands = self.engine.convert(context, keys, prefix=prefix, **self.quick)
        self._remember(context, keys, cands)
        return cands

    def _remember(self, context, keys, cands):
        if cands:
            self._last = (context, keys, self.engine.paths[cands[0]])
