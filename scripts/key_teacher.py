"""The tokens the keys still allow at each target position, for train.py --key-teacher (7B.1).

r6 learned jige -> 几个 as almost certain (及 −11.7 nats after 「只是到了」, rank 173): the plain loss only ever sees the
one target, so it never learns how the other readings of the same keys share the mass in a context. The key teacher is
the base model after the context and the target so far (it knows 及格 after 到了), normalised over only the tokens the
keys can still read from where the target token starts: a token of Han characters, each a non-empty prefix of one of
its readings (EXTRA_READINGS too), as the engine's trie reads them, ending where the rest of the keys can still be read,
with the engine's costs for an abbreviated character (ABBREV, plus SPLIT when its keys read one key further or more are
a whole syllable: j of jigexian is not 件) taken off the base model's scores. Unlike the
--teacher of r4 / D-015 (the fine-tuned model mixed with the base, i.e. the engine's own score, which is as sure as
the student), this one puts the mass where the context puts it.

Rows whose target the keys do not read this way (a typo, English in the target, a rare character split into byte
tokens) get no key-teacher term; nor does the end of sentence.
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import fmt  # noqa: E402
from ime.ngram import char_readings_table  # noqa: E402

MAX_COMBOS = 16  # as ime.llm
ABBREV, SPLIT = 1.0, 9.0  # v02's abbrev_penalty and split_penalty (ime.engine)
MAX_ABBREV = 4  # abbreviated characters a token's entry records (more: the token is left out)


class KeyMask:
    def __init__(self, tok):
        self.tok = tok
        self.readings = char_readings_table(extra=True)
        self.syllables = {r for rs in self.readings.values() for r in rs}
        self.prefixes = {r[:L] for r in self.syllables for L in range(1, len(r) + 1)}
        self.text = {}  # token id -> its text, Han tokens only
        pats = {}  # keys -> [(token id, ((offset, length) of each abbreviated character, ...))]
        for t, i in tok.get_vocab().items():
            s = tok.convert_tokens_to_string([t])
            if not s or not all(c in self.readings for c in s):
                continue
            self.text[i] = s
            for combo in itertools.islice(itertools.product(*(self.readings[c] for c in s)), MAX_COMBOS):
                for cut in itertools.product(*(range(1, len(r) + 1) for r in combo)):
                    ab, at = [], 0
                    for r, L in zip(combo, cut):
                        if L < len(r):
                            ab.append((at, L))
                        at += L
                    if len(ab) <= MAX_ABBREV:
                        pats.setdefault("".join(r[:L] for r, L in zip(combo, cut)), []).append((i, tuple(ab)))
        self.pats = {}
        for k, v in pats.items():
            v = sorted(set(v))
            off = np.full((len(v), MAX_ABBREV), -1, dtype=np.int16)
            ln = np.zeros((len(v), MAX_ABBREV), dtype=np.int16)
            for j, (_, ab) in enumerate(v):
                for m, (o, L) in enumerate(ab):
                    off[j, m], ln[j, m] = o, L
            self.pats[k] = (np.array([i for i, _ in v], dtype=np.int64), off, ln)
        self.max_len = max(map(len, self.pats))

    def _keys_info(self, keys):
        """Per key position: whether the keys from there can be read at all, and the longest syllable they start."""
        n = len(keys)
        ok = [False] * n + [True]
        for q in range(n - 1, -1, -1):
            ok[q] = any(ok[q + L] for L in range(1, min(6, n - q) + 1) if keys[q : q + L] in self.prefixes)
        longest = np.array([max([L for L in range(1, min(6, n - q) + 1) if keys[q : q + L] in self.syllables] or [0])
                            for q in range(n)] + [0], dtype=np.int16)
        return ok, longest

    def allowed(self, keys, e, info=None):
        """(token ids, costs) of the tokens that read keys from e to a position the rest of the keys can be read from,
        the lowest cost per token."""
        ok, longest = info or self._keys_info(keys)
        ids, costs = [], []
        for L in range(1, min(self.max_len, len(keys) - e) + 1):
            p = self.pats.get(keys[e : e + L])
            if p is None or not ok[e + L]:
                continue
            tid, off, ln = p
            has = off >= 0
            split = has & (longest[e + np.maximum(off, 0)] > ln)
            ids.append(tid)
            costs.append((has * ABBREV + split * SPLIT).sum(1).astype(np.float32))
        if not ids:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        ids, costs = np.concatenate(ids), np.concatenate(costs)
        o = np.lexsort((costs, ids))
        ids, costs = ids[o], costs[o]
        first = np.r_[True, ids[1:] != ids[:-1]]
        return ids[first], costs[first]

    def keys_of(self, prompt_ids):
        """The keys in a ctx_first prompt (context SEP1 k e y s SEP2), or None."""
        text = self.tok.decode(list(prompt_ids))
        a, b = text.rfind(fmt.SEP1), text.rfind(fmt.SEP2)
        ks = text[a + 1 : b].replace(" ", "") if 0 <= a < b else ""
        return ks if ks.isascii() and ks.isalpha() and ks.islower() else None

    def _ends(self, s, starts, keys):
        for ch in s:
            starts = {e + L for e in starts for r in self.readings[ch] for L in range(1, len(r) + 1)
                      if keys[e : e + L] == r[:L]}
        return starts

    def starts(self, keys, target_ids):
        """Per target token, the key positions it may start at on a reading of the whole keys, or None when the keys
        do not read the target (each character a non-empty prefix of one of its readings)."""
        texts = [self.text.get(int(t)) for t in target_ids]
        if not all(texts):
            return None
        fwd = [{0}]  # key positions reachable after each token
        for s in texts:
            fwd.append(self._ends(s, fwd[-1], keys))
            if not fwd[-1]:
                return None
        if len(keys) not in fwd[-1]:
            return None
        good, out = {len(keys)}, []  # backward: only starts from which the rest of the target reads the rest
        for k in range(len(texts) - 1, -1, -1):
            good = {e0 for e0 in fwd[k] if self._ends(texts[k], {e0}, keys) & good}
            out.append(sorted(good))
        return out[::-1]

    def row(self, seq, plen):
        """(token ids, costs) per target position of one row (prompt + target + EOS), None where there is no
        key-teacher term (the end of sentence; all of them when the keys do not read the target)."""
        n_t = len(seq) - plen
        keys = self.keys_of(seq[:plen])
        st = self.starts(keys, seq[plen:-1]) if keys else None
        if st is None:
            return [None] * n_t
        info = self._keys_info(keys)
        out = []
        for s in st:
            a = [self.allowed(keys, e, info) for e in s]
            if len(a) > 1:  # rare: the token may start at more than one key; the lowest cost per token
                ids, costs = np.concatenate([x[0] for x in a]), np.concatenate([x[1] for x in a])
                o = np.lexsort((costs, ids))
                ids, costs = ids[o], costs[o]
                first = np.r_[True, ids[1:] != ids[:-1]]
                a = [(ids[first], costs[first])]
            out.append(a[0])
        return out + [None]
