"""LLM converter: a base causal LM continues the left context, constrained to the typed keys.

Every vocab token made only of Han characters is indexed in a trie by its characters' readings
(same matching rule as the n-gram engine: each character consumes a non-empty prefix of a reading).
Beam search runs all beams as one batch, one token per step, reusing the prefilled context cache.
"""

import itertools
import os
import pickle
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from ime.ngram import Node, char_readings_table

MAX_COMBOS = 16  # reading combinations per token (polyphones multiply)


class LLMEngine:
    name = "llm"

    def __init__(self, model="models/qwen3-0.6b-base-bf16", beam=6, abbrev_penalty=1.0, empty_prompt="\n"):
        self.model, self.tok = load(model)
        self.beam, self.alpha, self.empty_prompt = beam, abbrev_penalty, empty_prompt
        self.root, self.tok_text = self._token_trie(model)
        self._ctx_key, self._ctx_cache, self._ctx_logits = None, None, None
        self.stats = {"forward_ms": 0.0, "calls": 0}

    # --- token index ----------------------------------------------------------
    def _token_trie(self, model):
        path = os.path.join(model, "han_tokens.pkl")
        if os.path.exists(path):
            entries = pickle.load(open(path, "rb"))
        else:
            chars = char_readings_table()
            entries = []
            vocab = self.tok._tokenizer.get_vocab()
            for t, i in vocab.items():
                s = self.tok._tokenizer.convert_tokens_to_string([t])
                if s and all(c in chars for c in s):
                    combos = list(itertools.islice(itertools.product(*(chars[c] for c in s)), MAX_COMBOS))
                    entries.append((i, s, combos))
            pickle.dump(entries, open(path, "wb"))
        root, text = Node(), {}
        for i, s, combos in entries:
            text[i] = s
            for sylls in combos:
                n = root
                for sy in sylls:
                    n = n.kids.setdefault(sy, Node())
                n.words.append(i)

        def finish(n):
            for sy, k in n.kids.items():
                n.by_letter.setdefault(sy[0], []).append(sy)
                finish(k)

        finish(root)
        return root, text

    def edges(self, keys, p):
        """{(end, token_id): fewest abbreviated chars} for tokens starting at key position p."""
        out = {}

        def walk(node, q, ab):
            for t in node.words:
                if out.get((q, t), 99) > ab:
                    out[(q, t)] = ab
            if q >= len(keys):
                return
            for sy in node.by_letter.get(keys[q], ()):
                kid = node.kids[sy]
                m = 1
                while m < len(sy) and q + m < len(keys) and keys[q + m] == sy[m]:
                    m += 1
                for L in range(1, m + 1):
                    walk(kid, q + L, ab + (L < len(sy)))

        walk(self.root, p, 0)
        ends = np.array([e for e, _ in out], dtype=np.int32)
        tids = np.array([t for _, t in out], dtype=np.int32)
        abs_ = np.array(list(out.values()), dtype=np.float32)
        return ends, tids, abs_

    # --- model ----------------------------------------------------------------
    def _prefill(self, context):
        prompt = context[-64:] if context.strip() else self.empty_prompt
        if prompt == self._ctx_key:
            return
        ids = mx.array(self.tok.encode(prompt))[None]
        cache = make_prompt_cache(self.model)
        logits = self.model(ids, cache=cache)[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
        self._ctx_key, self._ctx_cache, self._ctx_logits = prompt, cache, logits

    def _fork_cache(self, idx):
        """Per-beam copy of the context cache (rows idx of a batch-1 cache)."""
        cache = make_prompt_cache(self.model)
        for c, src in zip(cache, self._ctx_cache):
            k, v = src.state
            c.keys, c.values, c.offset = k[idx], v[idx], src.offset
        return cache

    @staticmethod
    def _reorder(cache, idx):
        for c in cache:
            c.keys = c.keys[idx, :, : c.offset]
            c.values = c.values[idx, :, : c.offset]

    def convert(self, context, keys, k=5):
        if not keys:
            return []
        self._prefill(context)
        n = len(keys)
        edge_cache = {}

        def edges_at(p):
            if p not in edge_cache:
                edge_cache[p] = self.edges(keys, p)
            return edge_cache[p]

        logits = self._ctx_logits
        active = [(0.0, "", 0)]  # (score, text, key position)
        cache = None
        finished = []
        for _step in range(n):
            lp = (logits - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.float32)
            lp = np.array(lp)
            cand_s, cand_parent, cand_tid, cand_end = [], [], [], []
            for b, (score, _text, p) in enumerate(active):
                ends, tids, abs_ = edges_at(p)
                if len(tids) == 0:
                    continue
                cand_s.append(score + lp[b, tids] - self.alpha * abs_)
                cand_parent.append(np.full(len(tids), b))
                cand_tid.append(tids)
                cand_end.append(ends)
            if not cand_s:
                break
            S, P, T, E = (np.concatenate(x) for x in (cand_s, cand_parent, cand_tid, cand_end))
            order = np.argsort(-S)
            nxt, parents, tids = [], [], []
            for j in order:
                text = active[P[j]][1] + self.tok_text[int(T[j])]
                if E[j] == n:
                    finished.append((float(S[j]), text))
                elif len(nxt) < self.beam:
                    nxt.append((float(S[j]), text, int(E[j])))
                    parents.append(int(P[j]))
                    tids.append(int(T[j]))
                if len(nxt) >= self.beam and len(finished) >= k * 3:
                    break
            finished.sort(reverse=True)
            if not nxt or (len(finished) >= k and finished[k - 1][0] >= nxt[0][0]):
                break
            idx = mx.array(parents)
            if cache is None:
                cache = self._fork_cache(idx)
            else:
                self._reorder(cache, idx)
            t0 = time.perf_counter()
            logits = self.model(mx.array(tids)[:, None], cache=cache)[:, -1, :]
            mx.eval(logits)
            self.stats["forward_ms"] += (time.perf_counter() - t0) * 1000
            self.stats["calls"] += 1
            active = nxt
        out, seen = [], set()
        for _s, t in sorted(finished, reverse=True):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out[:k]
