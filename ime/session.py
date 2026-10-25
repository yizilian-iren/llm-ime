"""Typing session: converts in a worker thread, quick guess first, full search after, stale work dropped.

Every keystroke calls `type(context, keys, tag=None)`. The tag says who is typing (the engine service gives
each connection its own); only a newer request with the same tag makes an older one stale, so two typists
do not starve each other. The worker takes each tag's newest request in turn and for it:
  1. quick: a narrow beam (top 3), with the mixed-in model too, since without it the quick guess agrees with the
     full search far less often (3A.2: 63% vs 40% of keys). Keys as short as people type them at once (under
     `prefix_keys`; on M3 94% of commits are 8 keys or fewer) are searched whole, scoring the end of the sentence
     like the full search: the quick guess's first is the full search's first on 95% of keys, not 86% (5B.10:
     starting from the last best and not scoring the end were most of the difference, a wider beam hardly any),
     for 10-20% more time. Longer keys start from the best candidate it last had for a prefix of these keys
     (minus that candidate's last token, whose reading the new key may change), so only the tail is searched,
     without the extra step that scores the end (both cut its time by about a quarter there);
  2. full: the engine's normal search (k candidates); dropped between beam steps as soon as a newer keystroke
     arrives. It starts `full_delay` s after the quick guess is sent, and not at all if a newer keystroke comes
     meanwhile: one beam step's forward cannot be stopped, and on M3 the next key came while the previous key's
     full search ran on 63% of keys from 4 on, its quick guess waiting for that forward (5B.11, 5B.12). `hurry(tag)`
     (space waiting for the full answer) starts it at once. Default 0: on M3 (scripts/delay_bench.py, the user's
     key times replayed) 40 / 70 ms left the quick guess of 4-8 keys at 78 ms p50 (the search itself; 79 with
     none), while the full answer after the last key came 63 / 95 ms later and space waited for it 3x as often;
  3. more (when `more` > k, or with `segments`): after a pause of `more_delay` s with nothing else to do, the
     search again for `more` candidates, sent as the full answer's k followed by the new ones (so the first
     page stays put); dropped as soon as any request arrives. Typing is not slowed by it (k=20 for the full
     search was: 4A.5). With `segments`, it also carries the engine's candidates for the start of the keys
     (engine.segments: [(text, keys used)], 4A.10) as `segments=`, leaving out those that use all the keys
     and are already among the whole candidates.
One pass (`two_pass` False, the default since 7D.2, D-050): only the quick guess is computed for each key, and space
uses what is on screen, nothing to wait for; the full answer is not computed at all. The pause's "more" still
comes, but after the quick guess's own candidates (the first page does not change), and a word the keys are the
whole pinyin of (6B.7) that is not among them comes right after them. `set_two_pass(True)` goes back to the two passes.
With `words` (ime.userwords.UserWords), the full answer is reranked by the user's words and their pieces for the
start of the keys come first (5A.3); the quick guess puts words learned for all of the keys first (liji 力积 shows
while typing), without the model call of the rerank (5B.10: that cost M3 54 -> 75 ms on the keys it applied to).
Keys that cannot all be read as pinyin (a typo) still get candidates: the longest prefix that can is
converted and the rest of the letters follow as typed (4A.4).
`type(..., english=True)`: the English mode's keys, answered once with stage "english" (6B.2, `_english`).
`on_result(context, keys, candidates, stage)` is called from the worker thread with stage "quick" / "full" / "more",
with `tag=` too when the request had one, and `segments=` with "more" when `segments` is on. An exception while converting goes to `on_error()` (called in
the except block; default: print the traceback) and the worker goes on with the next request.
All model work stays on the worker thread (MLX state is not shared between threads).
"""

import threading
import traceback

from ime import zh_en


class Session:
    def __init__(self, engine, on_result, k=5, quick=None, on_error=None, more=None, more_delay=0.25, segments=False,
                 words=None, prefix_keys=10, full_delay=0.0, two_pass=False, prefix_rows=3, prefix_merge=True):
        self.engine, self.on_result, self.k, self.words = engine, on_result, k, words
        self.more, self.more_delay = more, more_delay
        self.segments = segments and hasattr(engine, "segments")
        self.on_error = on_error or traceback.print_exc
        self.quick = quick or dict(k=3, beam=4, mix=True, eos=False)  # convert() options of the quick guess (beam 4: 7D.3, D-065)
        self.prefix_keys = prefix_keys  # keys this long or longer: the quick guess starts from the last best
        self.prefix_rows = prefix_rows  # 10 keys or more: the quick guess goes on from this many of the last paths (7D.3)
        self.prefix_merge = prefix_merge  # ...as one beam from the paths' common start, not a search from each (7D.3)
        self.full_delay = full_delay
        self.two_pass = two_pass
        self._cond = threading.Condition()
        self._pending = {}  # tag -> (context, keys), the newest request of each; first come, first served
        self._hurry = set()  # tags whose latest request's full search is wanted now
        self._closed = False
        self._last = None  # (context, keys, best path) of the latest result, quick or full
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def type(self, context, keys, tag=None, english=False):
        with self._cond:
            self._pending[tag] = (context, keys, english)  # an older one of this tag keeps its place in line
            self._hurry.discard(tag)
            self._cond.notify()

    def set_two_pass(self, two):
        self.two_pass = bool(two)  # read at the next key

    def hurry(self, tag=None):
        """The full answer to `tag`'s latest request is waited for (space): no `full_delay` before it."""
        with self._cond:
            self._hurry.add(tag)
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
                context, keys, english = self._pending.pop(tag)
            try:
                if english:
                    self._english(context, keys, tag)
                else:
                    self._serve(context, keys, tag)
            except Exception:
                self.on_error()

    def _serve(self, context, keys, tag):
        send = (lambda *a, **kw: self.on_result(*a, tag=tag, **kw)) if tag is not None else self.on_result
        if not keys:
            send(context, keys, [], "full")
            return
        quick = self._quick(context, keys)
        send(context, keys, quick, "quick")
        if not self.two_pass:
            self._more_after_quick(context, keys, quick, tag, send)
            return
        with self._cond:
            self._cond.wait_for(lambda: self._pending or self._closed or tag in self._hurry, timeout=self.full_delay)
            self._hurry.discard(tag)
            if tag in self._pending or self._closed:
                return
        full = self._convert(context, keys, k=self.k, stop=lambda: self._newer(tag))
        if full is None:
            return
        self._remember(context, keys, full)
        if self.words:
            full = self.words.rerank(context, keys, full)
        send(context, keys, full, "full")
        more = bool(self.more) and self.more > self.k and len(full) >= self.k
        if not more and not self.segments:
            return
        with self._cond:
            self._cond.wait_for(lambda: self._pending or self._closed, timeout=self.more_delay)
            if self._pending or self._closed:
                return  # someone is typing: the extra pages can wait
        whole = full
        if more:
            extra = self._convert(context, keys, k=self.more, stop=self._busy)
            if extra is None:
                return
            whole = full + [c for c in extra if c not in full]
        if not self.segments:
            send(context, keys, whole, "more")
            return
        if self._busy():
            return
        segs = self.engine.segments(context, keys)
        if self.words:
            segs = self.words.segments(keys, segs)
        segs = [(t, e) for t, e in segs if e < len(keys) or t not in whole]
        send(context, keys, whole, "more", segments=segs)

    def _more_after_quick(self, context, keys, quick, tag, send):
        """One pass: after a pause the pages behind the quick guess's own candidates (the search for `more`
        candidates), the whole-word candidate right after them; sent as "more", with the segments if wanted."""
        if not quick or not ((self.more and self.more > self.k) or self.segments):
            return
        with self._cond:
            self._cond.wait_for(lambda: self._pending or self._closed, timeout=self.more_delay)
            if self._pending or self._closed:
                return
        whole = quick
        if self.more and self.more > self.k:
            extra = self._convert(context, keys, k=self.more, stop=self._busy)
            if extra is None:
                return
            rest = [c for c in extra if c not in quick]
            sylls = getattr(self.engine, "words", None)
            named = {w for w, _s in sylls.get(keys, ())} if sylls and hasattr(sylls, "get") else set()
            word = next((c for c in rest if c in named), None) if named and not any(c in named for c in quick) else None
            if word:
                rest.remove(word)
                rest.insert(0, word)
            whole = quick + rest
        if not self.segments:
            send(context, keys, whole, "more")
            return
        if self._busy():
            return
        segs = self.engine.segments(context, keys)
        if self.words:
            segs = self.words.segments(keys, segs)
        segs = [(t, e) for t, e in segs if e < len(keys) or t not in whole]
        send(context, keys, whole, "more", segments=segs)

    def _english(self, context, keys, tag):
        """The English mode's keys (6B.2): when they are whole syllables of pinyin, the English of what they read as
        Chinese (pingguo: apple), one answer with stage "english"; none otherwise (the shell has the spelling)."""
        send = (lambda *a, **kw: self.on_result(*a, tag=tag, **kw)) if tag is not None else self.on_result
        cands = []
        whole = getattr(self.engine, "whole_pinyin", None)
        if whole and whole(keys):
            # English before it is no help; the first three only (xuexi: 学习, not 血洗's reeking of blood)
            chinese = self.engine.convert("", keys, k=3, stop=lambda: self._newer(tag))
            if chinese is None:
                return
            cands = zh_en.english(chinese)
        send(context, keys, cands, "english")

    def _busy(self):
        with self._cond:
            return bool(self._pending) or self._closed

    def _quick(self, context, keys):
        if len(keys) < self.prefix_keys:
            cands = self._convert(context, keys, **{**self.quick, "eos": True})
        else:
            prefix, prefixes = None, None
            if self._last and self._last[0] == context and keys.startswith(self._last[1]) and self._last[2]:
                prefix = list(self._last[2][:-1]) or None
                if self.prefix_rows > 1 and prefix:
                    paths = [list(q[:-1]) for q in self._last[3][: self.prefix_rows] if q[:-1]]
                    if self.prefix_merge:  # one beam: from where the paths part, its rows are the ways on
                        common = []
                        for step in zip(*paths):
                            if any(x != step[0] for x in step):
                                break
                            common.append(step[0])
                        prefix = common or None
                    else:
                        prefixes = []
                        for q in paths:
                            if q not in prefixes:
                                prefixes.append(q)
                        prefix = None if len(prefixes) < 2 else prefix
                        prefixes = prefixes if len(prefixes) > 1 else None
            cands = self._convert(context, keys, prefix=prefix, **({"prefixes": prefixes} if prefixes else {}), **self.quick)
        self._remember(context, keys, cands)
        if self.words and cands:
            cands = self.words.whole(keys, cands)
        return cands

    def _convert(self, context, keys, **kw):
        """engine.convert, or for keys it cannot read: the longest readable prefix converted + the rest as typed
        (after the keys themselves when the engine gave just those: an English word, github, 5B.9)."""
        cands = self.engine.convert(context, keys, **kw)
        if cands is None or (cands and cands != [keys]):
            return cands
        n = self._readable(keys)
        if n == 0:
            return [keys]
        head = self.engine.convert(context, keys[:n], **{**{a: b for a, b in kw.items() if a != "prefixes"}, "prefix": None})
        if head is None:
            return None
        return (cands + [c + keys[n:] for c in head if c + keys[n:] != keys])[: kw.get("k", 5)] or [keys]

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
            self._last = (context, keys, path, list(getattr(self.engine, "paths", {}).values()))
