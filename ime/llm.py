"""LLM converter: a base causal LM continues the left context, constrained to the typed keys.

Every vocab token made only of Han characters is indexed in a trie by its characters' readings
(same matching rule as the n-gram engine: each character consumes a non-empty prefix of a reading).
Beam search runs all beams as one batch, one token per step, reusing the prefilled context cache.
Optionally a second model (`mix`, e.g. the base model next to a fine-tuned one) scores the same beams
from the plain context; its log-probs are added with weight `mix_weight` (the end-of-sentence score
stays the first model's).
"""

import itertools
import os
import pickle
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from ime import fmt
from ime.ngram import Node, char_readings_table

MAX_COMBOS = 16  # reading combinations per token (polyphones multiply)


class LLMEngine:
    name = "llm"

    def __init__(self, model="models/qwen3-0.6b-base-4bit", beam=12, abbrev_penalty=1.0, progress_bonus=3.0,
                 empty_prompt="\n", keys_in_prompt=None, mix="", mix_weight=0.5, mix_weight_empty=None,
                 split_penalty=0.0):
        self.model, self.tok = load(model)
        self.mix = _Cached(load(mix)[0]) if mix else None
        # With no context the mixed-in model is only guessing: it gets its own (usually smaller) weight.
        self.mix_weight = mix_weight
        self.mix_weight_empty = mix_weight if mix_weight_empty is None else mix_weight_empty
        # Fine-tuned models see the keys in the prompt (ime.fmt) and score the end of the sentence;
        # base models just continue the context.
        self.order = fmt.model_order(model) or "ctx_first"
        if keys_in_prompt is None:
            keys_in_prompt = fmt.model_order(model) is not None
        self.keys_in_prompt = bool(int(keys_in_prompt))
        self.eos = self.tok.eos_token_id
        # progress_bonus: nats credited per key consumed when ranking beams, so hypotheses that have
        # consumed more of the input are not out-ranked just for having paid for more characters.
        self.beam, self.alpha, self.beta, self.empty_prompt = beam, abbrev_penalty, progress_bonus, empty_prompt
        self.prefixes = {r[:i] for rs in char_readings_table().values() for r in rs for i in range(1, len(r) + 1)}
        # split_penalty: nats more for an abbreviated character whose keys, read one key further or more,
        # are a whole syllable (ma read as 命啊 / 没啊 instead of 吗); real initials (jt of 今天) never pay it.
        self.syllables = {r for rs in char_readings_table().values() for r in rs}
        self.split_penalty = float(split_penalty)
        self.root, self.tok_text = self._token_trie(model)
        self._ids, self._ctx_cache, self._ctx_logits = [], None, None
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
        """{(end, token_id): lowest cost} for tokens starting at key position p: abbrev_penalty per abbreviated
        char, plus split_penalty when its keys are cut short of a whole syllable the user typed."""
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
                    cost = 0.0
                    if L < len(sy):
                        cost = self.alpha
                        if self.split_penalty and any(keys[q:j] in self.syllables for j in range(q + L + 1, min(q + 7, len(keys) + 1))):
                            cost += self.split_penalty
                    walk(kid, q + L, ab + cost)

        walk(self.root, p, 0.0)
        ends = np.array([e for e, _ in out], dtype=np.int32)
        tids = np.array([t for _, t in out], dtype=np.int32)
        abs_ = np.array(list(out.values()), dtype=np.float32)
        return ends, tids, abs_

    # --- model ----------------------------------------------------------------
    def _prefill(self, prompt):
        """Bring the batch-1 cache to `prompt`, reusing the longest common token prefix of the last one."""
        ids = self.tok.encode(prompt)
        if ids == self._ids:
            return
        common = 0
        for a, b in zip(ids, self._ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)  # always feed at least one token to get fresh logits
        if self._ctx_cache is None or common == 0:
            self._ctx_cache, common = make_prompt_cache(self.model), 0
        else:
            for c in self._ctx_cache:
                c.trim(c.offset - common)
        logits = self.model(mx.array(ids[common:])[None], cache=self._ctx_cache)[:, -1, :]
        mx.eval(logits, [c.state for c in self._ctx_cache])
        self._ids, self._ctx_logits = ids, logits

    def _fork_cache(self, idx, steps):
        return _fork(self.model, self._ctx_cache, idx, steps)

    @staticmethod
    def _reorder(cache, idx):
        for c in cache:
            c.keys = c.keys[idx]
            c.values = c.values[idx]

    def convert(self, context, keys, k=5, beam=None, mix=True, prefix=None, stop=None, eos=True):
        """Candidates for `keys`, best first; their token paths are left in `self.paths` (text -> path).

        beam: overrides the beam width. mix: False skips the mixed-in model.
        prefix: a path [(token_id, key_end), ...] to start from instead of the empty sentence (fed in one
          forward per model); for the quick first guess, the previous keystroke's best candidate minus its
          last token.
        stop: called between beam steps; when it returns True the search is dropped and None returned.
        eos: False finishes a hypothesis as soon as it has consumed all keys, without the extra step that
          scores the end of the sentence (fine-tuned models only; base models never score it).
        """
        self.paths = {}
        if not keys:
            return []
        beam = beam or self.beam
        mix = bool(self.mix) and mix
        plain = context[-64:] if context.strip() else self.empty_prompt
        if self.keys_in_prompt:
            self._prefill(fmt.prompt(context, keys, self.order))
        else:
            self._prefill(plain)
        if mix:
            self.mix.prefill(self.tok.encode(plain))
        n = len(keys)
        # feasible[p]: keys[p:] can be split into reading prefixes (else that hypothesis is a dead end)
        feasible = [False] * n + [True]
        for p in range(n - 1, -1, -1):
            feasible[p] = any(feasible[p + L] for L in range(1, min(6, n - p) + 1) if keys[p : p + L] in self.prefixes)
        if not feasible[0]:
            return []
        edge_cache = {}

        def edges_at(p):
            if p not in edge_cache:
                ends, tids, abs_ = self.edges(keys, p)
                ok = np.array([feasible[e] for e in ends], dtype=bool)
                edge_cache[p] = (ends[ok], tids[ok], abs_[ok]) if len(ends) else (ends, tids, abs_)
            return edge_cache[p]

        logits = self._ctx_logits
        mix_logits = self.mix.ctx_logits if mix else None
        mix_w = (self.mix_weight if context.strip() else self.mix_weight_empty) if mix else 0
        active = [(0.0, "", 0, ())]  # (score, text, key position, path)
        cache = mix_cache = None
        if prefix and prefix[-1][1] <= n and feasible[prefix[-1][1]]:
            path = tuple(prefix)
            cache = self._fork_cache(mx.array([0]), len(path) + n + 2)
            ptoks = mx.array([t for t, _e in path])[None]
            logits = self.model(ptoks, cache=cache)[:, -1, :]
            if mix_w:
                mix_cache = self.mix.fork(mx.array([0]), len(path) + n + 2)
                mix_logits = self.mix.model(ptoks, cache=mix_cache)[:, -1, :]
            active = [(0.0, "".join(self.tok_text[t] for t, _e in path), path[-1][1], path)]
        finished = []
        for _step in range(n + 1):
            if stop and stop():
                return None
            # Only the columns this step can use (the tokens that fit at the beams' key positions, and EOS)
            # leave the GPU: copying the whole vocab per beam cost more than the forward.
            pos = {p for _s, _t, p, _path in active if p < n}
            cols = np.unique(np.concatenate([edges_at(p)[1] for p in pos] + [np.array([self.eos], np.int32)]))
            col_of = lambda t: np.searchsorted(cols, t)  # noqa: E731
            lp = _log_softmax_cols(logits, mx.array(cols))
            if mix_w:
                mlp = _log_softmax_cols(mix_logits, mx.array(cols))
                mx.eval(lp, mlp)
                lp, mlp = np.array(lp), np.array(mlp)
                eos_lp = lp[:, col_of(self.eos)].copy()
                lp = lp + mix_w * mlp
                lp[:, col_of(self.eos)] = eos_lp
            else:
                lp = np.array(lp)
            cand_s, cand_parent, cand_tid, cand_end = [], [], [], []
            for b, (score, text, p, path) in enumerate(active):
                if p == n:  # all keys consumed, waiting for its end-of-sentence score
                    finished.append((score + float(lp[b, col_of(self.eos)]), text, path))
                    continue
                ends, tids, abs_ = edges_at(p)
                if len(tids) == 0:
                    continue
                cand_s.append(score + lp[b, col_of(tids)] - abs_)
                cand_parent.append(np.full(len(tids), b))
                cand_tid.append(tids)
                cand_end.append(ends)
            if not cand_s:
                finished.sort(reverse=True)
                break
            S, P, T, E = (np.concatenate(x) for x in (cand_s, cand_parent, cand_tid, cand_end))
            order = np.argsort(-(S + self.beta * E))
            nxt, parents, tids = [], [], []
            for j in order:
                text = active[P[j]][1] + self.tok_text[int(T[j])]
                path = active[P[j]][3] + ((int(T[j]), int(E[j])),)
                if E[j] == n and not (self.keys_in_prompt and eos):
                    finished.append((float(S[j]), text, path))
                elif len(nxt) < beam:
                    nxt.append((float(S[j]), text, int(E[j]), path))
                    parents.append(int(P[j]))
                    tids.append(int(T[j]))
                if len(nxt) >= beam and len(finished) >= k * 3:
                    break
            finished.sort(reverse=True)
            if not nxt or (len(finished) >= k and finished[k - 1][0] >= max(h[0] for h in nxt)):
                break
            idx = mx.array(parents)
            if cache is None:
                cache = self._fork_cache(idx, n + 2)
            else:
                self._reorder(cache, idx)
            t0 = time.perf_counter()
            logits = self.model(mx.array(tids)[:, None], cache=cache)[:, -1, :]
            if mix_w:
                mix_cache = self.mix.fork(idx, n + 2) if mix_cache is None else (self._reorder(mix_cache, idx) or mix_cache)
                mix_logits = self.mix.model(mx.array(tids)[:, None], cache=mix_cache)[:, -1, :]
                mx.eval(logits, mix_logits)
            else:
                mx.eval(logits)
            self.stats["forward_ms"] += (time.perf_counter() - t0) * 1000
            self.stats["calls"] += 1
            active = nxt
        out = []
        for _s, t, path in sorted(finished, reverse=True):
            if t not in self.paths:
                self.paths[t] = path
                out.append(t)
        return out[:k]


def _log_softmax_cols(logits, cols):
    """log_softmax(logits)[:, cols] as float32, computed as the whole-vocab version was (same rounding)."""
    return (logits[:, cols] - mx.logsumexp(logits, axis=-1, keepdims=True)).astype(mx.float32)


class _Cached:
    """A second model's batch-1 context cache, kept like LLMEngine's own (_prefill / _fork_cache)."""

    def __init__(self, model):
        self.model, self.ids, self.cache, self.ctx_logits = model, [], None, None

    def prefill(self, ids):
        if ids == self.ids:
            return
        common = 0
        for a, b in zip(ids, self.ids):
            if a != b:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if self.cache is None or common == 0:
            self.cache, common = make_prompt_cache(self.model), 0
        else:
            for c in self.cache:
                c.trim(c.offset - common)
        self.ctx_logits = self.model(mx.array(ids[common:])[None], cache=self.cache)[:, -1, :]
        mx.eval(self.ctx_logits, [c.state for c in self.cache])
        self.ids = ids

    def fork(self, idx, steps):
        return _fork(self.model, self.cache, idx, steps)


def _fork(model, ctx_cache, idx, steps):
    """Per-beam copy (rows idx) of a batch-1 context cache, sized for `steps` more tokens: every beam
    reorder copies the whole buffer, so it must not carry the default 256-slot preallocation."""
    cache = make_prompt_cache(model)
    for c, src in zip(cache, ctx_cache):
        k, v = src.keys, src.values
        off, cap = src.offset, src.offset + steps
        if k.shape[2] >= cap:
            k, v = k[..., :cap, :], v[..., :cap, :]
        else:
            pad = [(0, 0), (0, 0), (0, cap - k.shape[2]), (0, 0)]
            k, v = mx.pad(k, pad), mx.pad(v, pad)
        c.keys, c.values, c.offset = k[idx], v[idx], off
    return cache
