"""Interleave engine (r4b): the model reads each character right after its own keys (ime.fmt "interleave"), so
a hypothesis never depends on keys it has not consumed yet.

The search feeds one token per hypothesis per step. A hypothesis is either after a character (or at the start),
and its next token is forced: the next key; or after a key, and then it may take one more key (while the keys
since its last character are still the prefix of a reading) and/or pick a character whose reading starts with
those keys, scored by the model's log-prob there (minus abbrev_penalty when the keys are not the whole reading).
Only character picks compete for the beam; ranking adds progress_bonus per key consumed, as in ime.llm.
Characters are single tokens; the rare ones that are not are never offered.

Not incremental yet: every call searches the whole key string again (r4b's first step is to measure accuracy).
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from ime import fmt
from ime.llm import _fork, _log_softmax_cols
from ime.ngram import char_readings_table


class InterleaveEngine:
    name = "llm"

    def __init__(self, model, beam=12, abbrev_penalty=1.0, progress_bonus=3.0, **_ignored):
        self.model, self.tok = load(model)
        self.beam, self.alpha, self.beta = beam, abbrev_penalty, progress_bonus
        self.eos = self.tok.eos_token_id
        self.letter = {c: self.tok.encode(c)[0] for c in "abcdefghijklmnopqrstuvwxyz"}
        # reading prefix -> (token ids of the characters with a reading starting with it, 1 where not whole)
        by_prefix, self.text = {}, {}
        for ch, rs in char_readings_table().items():
            ids = self.tok.encode(ch)
            if len(ids) != 1:
                continue
            self.text[ids[0]] = ch
            for r in rs:
                for i in range(1, len(r) + 1):
                    d = by_prefix.setdefault(r[:i], {})
                    d[ids[0]] = min(d.get(ids[0], 1), int(i < len(r)))
        self.cands = {p: (np.array(list(d), np.int32), np.array(list(d.values()), np.float32))
                      for p, d in by_prefix.items()}
        self._ids, self._cache = [], None
        self.stats = {"forward_ms": 0.0, "calls": 0}

    def _prefill(self, ids):
        """Bring the batch-1 cache to `ids`, reusing the longest common prefix of the last ones."""
        if ids == self._ids:
            return
        common = 0
        for a, b in zip(ids, self._ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if self._cache is None or common == 0:
            self._cache, common = make_prompt_cache(self.model), 0
        else:
            for c in self._cache:
                c.trim(c.offset - common)
        mx.eval(self.model(mx.array(ids[common:])[None], cache=self._cache), [c.state for c in self._cache])
        self._ids = ids

    def convert(self, context, keys, k=5, beam=None, stop=None, eos=True, **_ignored):
        """Candidates for `keys`, best first. stop: called between steps, True drops the search (None
        returned). eos: False finishes a hypothesis when its last character consumes the last key, without
        the extra step that scores the end of the sentence."""
        self.paths = {}
        if not keys:
            return []
        beam, n = beam or self.beam, len(keys)
        feasible = [False] * n + [True]
        for p in range(n - 1, -1, -1):
            feasible[p] = any(feasible[p + L] for L in range(1, min(6, n - p) + 1) if keys[p : p + L] in self.cands)
        if not feasible[0]:
            return []
        self._prefill(fmt.interleave(self.tok.encode, context, [], "")[0])
        active = [(0.0, "", 0, 0, False)]  # (score, text, start of the current keys, key position, after a key)
        logits = cache = None
        finished = []
        for _step in range(2 * n + 2):
            if stop and stop():
                return None
            segs = {keys[s:q] for _sc, _t, s, q, ak in active if ak and feasible[q]}
            lp = col = None
            if segs or any(not ak and q == n for *_x, q, ak in active):
                cols = np.unique(np.concatenate([self.cands[g][0] for g in segs] + [np.array([self.eos], np.int32)]))
                col = lambda t: np.searchsorted(cols, t)  # noqa: E731
                lp = np.array(_log_softmax_cols(logits, mx.array(cols)))
            feed, parents, nxt = [], [], []
            picks = []  # (ranking score, score, parent, token, key position)
            for b, (score, text, s, q, ak) in enumerate(active):
                if not ak:
                    if q == n:
                        finished.append((score + float(lp[b, col(self.eos)]), text))
                    elif keys[q] in self.cands:
                        feed.append(self.letter[keys[q]]), parents.append(b), nxt.append((score, text, q, q + 1, True))
                    continue
                if feasible[q]:
                    ids, ab = self.cands[keys[s:q]]
                    sc = score + lp[b, col(ids)] - self.alpha * ab
                    picks += [(x + self.beta * q, x, b, int(t), q) for x, t in zip(sc, ids)]
                if q < n and keys[s : q + 1] in self.cands:
                    feed.append(self.letter[keys[q]]), parents.append(b), nxt.append((score, text, s, q + 1, True))
            picks.sort(key=lambda x: -x[0])
            for _r, sc, b, t, q in picks[:beam]:
                text = active[b][1] + self.text[t]
                if q == n and not eos:
                    finished.append((float(sc), text))
                else:
                    feed.append(t), parents.append(b), nxt.append((float(sc), text, q, q, False))
            finished.sort(reverse=True)
            if not feed or (len(finished) >= k and finished[k - 1][0] >= max(h[0] for h in nxt)):
                break
            idx = mx.array(parents)
            if cache is None:
                cache = _fork(self.model, self._cache, idx, 2 * n + 2)
            else:
                for c in cache:
                    c.keys, c.values = c.keys[idx], c.values[idx]
            t0 = time.perf_counter()
            logits = self.model(mx.array(feed)[:, None], cache=cache)[:, -1, :]
            mx.eval(logits)
            self.stats["forward_ms"] += (time.perf_counter() - t0) * 1000
            self.stats["calls"] += 1
            active = nxt
        out = []
        for _s, t in sorted(finished, reverse=True):
            if t not in self.paths:
                self.paths[t] = ()  # no token path to resume from (Session's quick guess starts from scratch)
                out.append(t)
        return out[:k]
