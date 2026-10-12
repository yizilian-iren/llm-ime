"""Interleave engine (r4b): the model reads each character right after its own keys (ime.fmt "interleave"), so
a hypothesis never depends on keys it has not consumed yet.

The search feeds one token per hypothesis per step. A hypothesis is either after a character (or at the start),
and its next token is forced: the next key; or after a key, and then it may take one more key (while the keys
since its last character are still the prefix of a reading) and/or pick a character whose reading starts with
those keys, scored by the model's log-prob there (minus abbrev_penalty when the keys are not the whole reading).
Only character picks compete for the beam; ranking adds progress_bonus per key consumed, as in ime.llm.
key_lp: weight of the model's log-prob of each key it is fed (a model trained with --key-weight scores the
keys too, so a reading split wrongly, or a character that the following keys do not fit, costs there).
mix: a second model (the base model, as in v02) rescores the finished sentences once, reading the plain
context and each sentence with its own tokenization, weight mix_weight (mix_weight_empty with no context).
Scoring it per character during the search does not work: fed one character per token, the base model
gives a character that it would normally write inside a longer token (还 of 还有) a poor log-prob.
Characters are single tokens; the rare ones that are not are never offered.

Not incremental yet: every call searches the whole key string again (r4b's first step is to measure accuracy).
"""

import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from ime import fmt
from ime.llm import _fork, _log_softmax_cols
from ime.ngram import char_readings_table


class InterleaveEngine:
    name = "llm"

    def __init__(self, model, beam=12, abbrev_penalty=1.0, progress_bonus=3.0, key_lp=1.0, mix="",
                 mix_weight=1.0, mix_weight_empty=0.5, empty_prompt="\n", **_ignored):
        self.model, self.tok = load(model)
        self.beam, self.alpha, self.beta, self.gamma = int(beam), float(abbrev_penalty), float(progress_bonus), float(key_lp)
        self.mix = load(mix)[0] if mix else None
        self.mix_w, self.mix_w_empty, self.empty_prompt = float(mix_weight), float(mix_weight_empty), empty_prompt
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

    def _mix_scores(self, context, texts):
        """The mix model's log-prob of each text after the plain context (one batched forward)."""
        plain = self.tok.encode(context[-64:] if context.strip() else self.empty_prompt)
        seqs = [self.tok.encode(t) for t in texts]
        T = max(map(len, seqs))
        x = np.zeros((len(seqs), len(plain) + T), np.int32)
        for r, s in enumerate(seqs):
            x[r, : len(plain)], x[r, len(plain) : len(plain) + len(s)] = plain, s
        t0 = time.perf_counter()
        lp = nn.log_softmax(self.mix(mx.array(x[:, :-1]))[:, len(plain) - 1 :, :].astype(mx.float32), axis=-1)
        tgt = mx.array(x[:, len(plain) :])
        got = np.array(mx.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0])
        self.stats["mix_ms"] = self.stats.get("mix_ms", 0.0) + (time.perf_counter() - t0) * 1000
        return [float(got[r, : len(s)].sum()) for r, s in enumerate(seqs)]

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
            if logits is not None:  # the first step's only hypothesis takes the first key: nothing to compare
                extra = np.array([self.eos] + ([self.letter[c] for c in set(keys) if c in self.letter]
                                               if self.gamma else []), np.int32)
                cols = np.unique(np.concatenate([self.cands[g][0] for g in segs] + [extra]))
                col = lambda t: np.searchsorted(cols, t)  # noqa: E731
                lp = np.array(_log_softmax_cols(logits, mx.array(cols)))
            feed, parents, nxt = [], [], []
            picks = []  # (ranking score, score, parent, token, key position)
            for b, (score, text, s, q, ak) in enumerate(active):
                if not ak:
                    if q == n:
                        finished.append((score + float(lp[b, col(self.eos)]), text))
                    elif keys[q] in self.cands:
                        sk = score + (self.gamma * float(lp[b, col(self.letter[keys[q]])]) if lp is not None and self.gamma else 0.0)
                        feed.append(self.letter[keys[q]]), parents.append(b), nxt.append((sk, text, q, q + 1, True))
                    continue
                if feasible[q]:
                    ids, ab = self.cands[keys[s:q]]
                    sc = score + lp[b, col(ids)] - self.alpha * ab
                    picks += [(x + self.beta * q, x, b, int(t), q) for x, t in zip(sc, ids)]
                if q < n and keys[s : q + 1] in self.cands:
                    sk = score + (self.gamma * float(lp[b, col(self.letter[keys[q]])]) if self.gamma else 0.0)
                    feed.append(self.letter[keys[q]]), parents.append(b), nxt.append((sk, text, s, q + 1, True))
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
        if self.mix and finished:
            best = {}
            for sc, t in finished:
                best[t] = max(best.get(t, sc), sc)
            top = sorted(best.items(), key=lambda x: -x[1])[:beam]
            mw = self.mix_w if context.strip() else self.mix_w_empty
            ms = self._mix_scores(context, [t for t, _ in top])
            finished = [(sc + mw * m, t) for (t, sc), m in zip(top, ms)]
        out = []
        for _s, t in sorted(finished, reverse=True):
            if t not in self.paths:
                self.paths[t] = ()  # no token path to resume from (Session's quick guess starts from scratch)
                out.append(t)
        return out[:k]
