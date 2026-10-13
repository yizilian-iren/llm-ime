"""Typing session: converts in a worker thread, quick guess first, full search after, stale work dropped.

Every keystroke calls `type(context, keys, tag=None)`. The tag says who is typing (the engine service gives
each connection its own); only a newer request with the same tag makes an older one stale, so two typists
do not starve each other. The worker takes each tag's newest request in turn and for it:
  1. quick: a narrow beam starting from the best candidate it last had for a prefix of these keys (minus
     that candidate's last token, whose reading the new key may change), so only the tail is searched;
     with the mixed-in model too, since without it the quick guess agrees with the full search far less
     often (3A.2: 63% vs 40% of keys); top 3 only, and without the extra step that scores the end of the
     sentence (both cut its time by about a quarter for 1 point of agreement);
  2. full: the engine's normal search; dropped between beam steps as soon as a newer keystroke arrives.
Keys that cannot all be read as pinyin (a typo) still get candidates: the longest prefix that can is
converted and the rest of the letters follow as typed (4A.4).
`on_result(context, keys, candidates, stage)` is called from the worker thread with stage "quick" / "full",
and with `tag=` too when the request had one. An exception while converting goes to `on_error()` (called in
the except block; default: print the traceback) and the worker goes on with the next request.
All model work stays on the worker thread (MLX state is not shared between threads).
"""

import threading
import traceback


class Session:
    def __init__(self, engine, on_result, k=5, quick=None, on_error=None):
        self.engine, self.on_result, self.k = engine, on_result, k
        self.on_error = on_error or traceback.print_exc
        self.quick = quick or dict(k=3, beam=3, mix=True, eos=False)  # convert() options of the quick guess
        self._cond = threading.Condition()
        self._pending = {}  # tag -> (context, keys), the newest request of each; first come, first served
        self._closed = False
        self._last = None  # (context, keys, best path) of the latest result, quick or full
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def type(self, context, keys, tag=None):
        with self._cond:
            self._pending[tag] = (context, keys)  # an older one of this tag keeps its place in line
            self._cond.notify()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify()
        self._thread.join()

    def _newer(self, tag):
        with self._cond:
            return tag in self._pending

    def _run(self):
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                tag = next(iter(self._pending))
                context, keys = self._pending.pop(tag)
            try:
                self._serve(context, keys, tag)
            except Exception:
                self.on_error()

    def _serve(self, context, keys, tag):
        send = (lambda *a: self.on_result(*a, tag=tag)) if tag is not None else self.on_result
        if not keys:
            send(context, keys, [], "full")
            return
        quick = self._quick(context, keys)
        send(context, keys, quick, "quick")
        if self._newer(tag):
            return
        full = self._convert(context, keys, k=self.k, stop=lambda: self._newer(tag))
        if full is None:
            return
        self._remember(context, keys, full)
        send(context, keys, full, "full")

    def _quick(self, context, keys):
        prefix = None
        if self._last and self._last[0] == context and keys.startswith(self._last[1]) and self._last[2]:
            prefix = list(self._last[2][:-1]) or None
        cands = self._convert(context, keys, prefix=prefix, **self.quick)
        self._remember(context, keys, cands)
        return cands

    def _convert(self, context, keys, **kw):
        """engine.convert, or for keys it cannot read: the longest readable prefix converted + the rest as typed."""
        cands = self.engine.convert(context, keys, **kw)
        if cands or cands is None:
            return cands
        n = self._readable(keys)
        if n == 0:
            return [keys]
        head = self.engine.convert(context, keys[:n], **{**kw, "prefix": None})
        if head is None:
            return None
        return [c + keys[n:] for c in head] or [keys]

    def _readable(self, keys):
        """Length of the longest prefix of `keys` that splits into reading prefixes (0 if the engine can't tell)."""
        prefixes = getattr(self.engine, "prefixes", None)
        if not prefixes:
            return 0
        ok = [True] + [False] * len(keys)
        for i in range(1, len(keys) + 1):
            ok[i] = any(ok[i - L] and keys[i - L : i] in prefixes for L in range(1, min(6, i) + 1))
        return max(i for i in range(len(keys) + 1) if ok[i])

    def _remember(self, context, keys, cands):
        path = cands and getattr(self.engine, "paths", {}).get(cands[0])
        if path:  # not for a candidate with typed letters at the end: it has no token path
            self._last = (context, keys, path)
