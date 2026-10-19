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
from ime.ngram import EXTRA_READINGS, Node, char_logp, char_readings_table, with_extra

MAX_COMBOS = 16  # reading combinations per token (polyphones multiply)
SINGLE_PAGE = 5  # one whole syllable typed: this many of its characters come before any other candidate
EN_WORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "en_words.txt")  # scripts/build_en_words.py
IDIOMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idioms.tsv")  # scripts/build_idioms.py
WORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "words.tsv")  # scripts/build_words.py
TRAD_CHARS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trad_chars.txt")  # scripts/build_trad_chars.py


class LLMEngine:
    name = "llm"

    def __init__(self, model="models/qwen3-0.6b-base-4bit", beam=12, abbrev_penalty=1.0, progress_bonus=3.0,
                 empty_prompt="\n", keys_in_prompt=None, mix="", mix_weight=0.5, mix_weight_empty=None,
                 split_penalty=0.0, swap_penalty=0.0, swap_margin=8.0,
                 swap_retries=1, alt_bonus=0.0, single_weight=0.0, english=False, typo_penalty=0.0,
                 typo_rows=3, english_penalty=0.0, english_pinyin_penalty=0.0, idiom_bonus=0.0,
                 word_top=0, word_slot=4, word_n=1):
        self.model, self.tok = load(model)
        self.model_path = model  # for the service's log: the base or the weekly retrain's (6A.2)
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
        self.char_readings = char_readings_table()
        self.char_readings_extra = char_readings_table(extra=True)
        self.syllables = {r for rs in self.char_readings.values() for r in rs}
        self.split_penalty = float(split_penalty)
        # swap_penalty: nats for reading two adjacent keys the other way round (zhognguo as 中国); 0 turns it off.
        # Both letters must fall in whole syllables of one token, so an abbreviation is never read as a swap.
        self.swap_penalty = float(swap_penalty)
        # The fine-tuned model reads the keys as typed and disbelieves a swap: a swap reading that, before its
        # swap_penalty, is within swap_margin nats of the best is searched again with the swapped keys in the
        # prompt (the best swap_retries of them), and those results replace the swap readings.
        self.swap_margin, self.swap_retries = float(swap_margin), int(swap_retries)
        # alt_bonus: nats credited per character read by EXTRA_READINGS (嗯 as en: the user writes 嗯 216 times, 恩 once)
        self.alt_bonus = float(alt_bonus)
        # single_weight: keys that are one whole syllable (wo, zai, hen) list its characters first, by how common
        # they are (ngram.char_freq) plus single_weight x the model's score; 0 leaves them to the search
        self.single_weight = float(single_weight)
        self.char_logp = char_logp() if self.single_weight else {}
        # english: keys that are an English word (ime/en_words.txt: github, cpu, tracker) and not pinyin, not even
        # pinyin typed halfway (long, fan, and bus on the way to bushi, stay Chinese) get the word as typed first
        self.english = _english_words() if english or english_penalty else frozenset()
        # english_penalty: nats for reading keys inside the sentence as an English word typed as it is (woyongpythonxie:
        # 我用python写, 6B.1), its tokens scored by the model like any other; 0 turns it off. english_pinyin_penalty: more
        # for a word whose letters are whole pinyin syllables too (change: 嫦娥, long: 龙).
        self.english_penalty, self.english_pinyin_penalty = float(english_penalty), float(english_pinyin_penalty)
        self._en_tokens = {}  # word -> ((token id, key end offset), ...)
        # idiom_bonus: nats credited for keys that are the whole pinyin of an idiom of ime/idioms.tsv (zhigenzhidi 知根知底,
        # 6B.3), its tokens scored by the model like any other; 0 turns it off
        self.idiom_bonus = float(idiom_bonus)
        self.idioms, self._idiom_lens = _idioms() if self.idiom_bonus else ({}, ())
        self._idiom_tokens = {}  # idiom or word -> ((token id, key end offset), ...) or ()
        # word_top: the keys that are the whole pinyin of one of the word_top most frequent words of ime/words.tsv
        # (jieba's dictionary, 6B.7) get up to word_n of them, the most frequent first, at place word_slot of the
        # candidates if the search has not found them: jigexian 及格线 is 20 nats behind 几个线 for the model, no bonus
        # that lifts it leaves the rest of the sentences alone (a floor on the first token, word edges: plain set -6 to
        # -28 points); 0 turns it off
        self.word_top, self.word_slot, self.word_n = int(word_top), int(word_slot), int(word_n)
        self.words = _words(self.word_top) if self.word_top else {}
        self.pinyin = {s for s in self.syllables if any(v in s for v in "aeiouv")}  # not n, m, hm of 嗯 呣
        self.pinyin_starts = {s[:i] for s in self.pinyin for i in range(1, len(s) + 1)}
        # typo_penalty: nats for reading a typed syllable with one key too many, one missing, or one hit next to the
        # right one (yiijing, fangkeshijao, hainebg; 5B.8); 0 turns it off. Like a swap: the syllable is read whole,
        # and is searched again with the keys fixed (swap_margin) for the fine-tuned model.
        self.typo_penalty, self.typo_rows = float(typo_penalty), int(typo_rows)
        self.typos = _typos(self.pinyin, self.prefixes, self.typo_penalty) if self.typo_penalty else {}
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
        for i, s, combos in entries:  # then EXTRA_READINGS (not in the cached entries), marked as such
            if not any(c in EXTRA_READINGS for c in s):
                continue
            for sylls in combos:
                for var in with_extra(s, list(sylls))[1:]:
                    n = root
                    for sy in var:
                        n = n.kids.setdefault(sy, Node())
                    if i not in n.words:
                        n.words.append(i)
                        n.alt[i] = tuple((j, std, sy) for j, (std, sy) in enumerate(zip(sylls, var)) if std != sy)

        def finish(n):
            for sy, k in n.kids.items():
                n.by_letter.setdefault(sy[0], []).append(sy)
                finish(k)

        finish(root)
        return root, text

    def edges(self, keys, p, swaps=True, alts=True):
        """Tokens starting at key position p: (ends, token ids, costs, fixes), the lowest cost per (end, token):
        abbrev_penalty per abbreviated char, plus split_penalty when its keys are cut short of a whole syllable
        the user typed. Also readings that do not follow the keys as the fine-tuned model learned them, each a fix
        (start, end, standard keys, cost) to the keys: with `swaps`, two adjacent keys inside the token read the
        other way round (swap_penalty); with `alts`, a character read by one of EXTRA_READINGS typed whole (嗯 as en,
        not e; -alt_bonus), never in a swapped token. fixes[i] is the token's tuple of them (empty: read as typed)."""
        out = {}
        n = len(keys)

        def walk(node, q, ab, ks, sw, spans, tf=None):
            # ks: the keys as read (keys, or keys with sw and sw + 1 exchanged); the token may end only past the swap.
            # spans: (start, length) of each syllable read so far in this token; tf: the token's typo fix, if any
            if sw is None or q >= sw + 2:
                fix = ((sw, sw + 2, ks[sw : sw + 2], self.swap_penalty),) if sw is not None else (tf,) if tf else ()
                for t in node.words:
                    alt, c = node.alt.get(t), ab
                    if alt:
                        if not alts or sw is not None or tf or any(spans[i][1] != len(sy) for i, _std, sy in alt):
                            continue
                        c -= self.alt_bonus * len(alt)
                        fix_t = tuple((spans[i][0], spans[i][0] + len(sy), std, -self.alt_bonus) for i, std, sy in alt)
                    else:
                        fix_t = fix
                    if out.get((q, t), (99,))[0] > c:
                        out[(q, t)] = (c, fix_t)
            if q >= n:
                return
            for sy in node.by_letter.get(ks[q], ()):
                kid = node.kids[sy]
                m = 1
                while m < len(sy) and q + m < n and ks[q + m] == sy[m]:
                    m += 1
                for L in range(1, m + 1):
                    if sw is not None and L < len(sy) and q <= sw + 1 and q + L > sw:
                        continue  # a swapped letter in an abbreviated syllable
                    cost = 0.0
                    if L < len(sy):
                        cost = self.alpha
                        if self.split_penalty and any(ks[q:j] in self.syllables for j in range(q + L + 1, min(q + 7, n + 1))):
                            cost += self.split_penalty
                    walk(kid, q + L, ab + cost, ks, sw, spans + ((q, L),), tf)
            if sw is None and tf is None and swaps and self.typo_penalty:
                # one syllable from q typed with one key too many, missing or next to the right one, read whole
                # (not keys a swap reads as a syllable from q: zheyagn is 这样 swapped, not yag 样 + n 呢)
                for m in range(max(2, swap_end(q) - q + 1), min(8, n - q + 1)):
                    for sy, cost in self.typos.get(keys[q : q + m], ()):
                        if sy in node.kids:
                            walk(node.kids[sy], q + m, ab + cost, ks, None, spans + ((q, m),), (q, q + m, sy, cost))
            if sw is None and tf is None and swaps and self.swap_penalty:
                # the swap starts in the syllable read from q (whole, swapped letters and all)
                for j in range(q, min(q + 6, n - 1)):
                    if keys[j] == keys[j + 1]:
                        continue
                    ks2 = keys[:j] + keys[j + 1] + keys[j] + keys[j + 2:]
                    for sy in node.by_letter.get(ks2[q], ()):
                        L = len(sy)
                        if q + L > j and ks2[q : q + L] == sy:
                            walk(node.kids[sy], q + L, ab + self.swap_penalty, ks2, j, spans + ((q, L),))

        swap_cache = {}

        def swap_end(q):  # where the longest syllable from q read with two adjacent keys in it exchanged ends (or q)
            if q not in swap_cache:
                swap_cache[q] = max([q] + [q + L for j in range(q, min(q + 6, n - 1)) if keys[j] != keys[j + 1]
                                           for L in range(j - q + 2, 7)
                                           if (keys[:j] + keys[j + 1] + keys[j] + keys[j + 2:])[q : q + L] in self.syllables]
                                    if self.swap_penalty else [q])
            return swap_cache[q]

        walk(self.root, p, 0.0, keys, None, ())
        ends = np.array([e for e, _ in out], dtype=np.int32)
        tids = np.array([t for _, t in out], dtype=np.int32)
        abs_ = np.array([c for c, _ in out.values()], dtype=np.float32)
        fixes_ = np.empty(len(out), dtype=object)
        fixes_[:] = [f for _, f in out.values()]
        return ends, tids, abs_, fixes_

    def english_edges(self, keys, p):
        """English words starting at key position p, typed as they are (english_penalty): (ends, token ids, costs, fixes,
        rests) like edges, the token being the word's first; rests: the word's other tokens [(token id, key end)], fed
        one a step before the hypothesis reads on."""
        out = []
        if self.english_penalty:
            for L in range(2, min(20, len(keys) - p) + 1):
                w = keys[p : p + L]
                if w not in self.english:
                    continue
                toks = self._en_tokens.get(w)
                if toks is None:
                    ids, toks, at = self.tok.encode(w), [], 0
                    for t in ids:
                        piece = self.tok.decode([t])
                        at += len(piece)
                        toks.append((t, at))
                        self.tok_text.setdefault(t, piece)
                    toks = self._en_tokens[w] = tuple(toks) if at == len(w) else ()
                if toks:
                    cost = self.english_penalty + (self.english_pinyin_penalty if self.whole_pinyin(w) else 0.0)
                    out.append((p + toks[0][1], toks[0][0], cost, tuple((t, p + e) for t, e in toks[1:])))
        ends = np.array([o[0] for o in out], dtype=np.int32)
        tids = np.array([o[1] for o in out], dtype=np.int32)
        costs = np.array([o[2] for o in out], dtype=np.float32)
        fixes_, rests = np.empty(len(out), dtype=object), np.empty(len(out), dtype=object)
        fixes_[:] = [()] * len(out)
        rests[:] = [o[3] for o in out]
        return ends, tids, costs, fixes_, rests

    def idiom_edges(self, keys, p):
        """Idioms of which the keys from position p are the whole pinyin (idiom_bonus): like english_edges, at cost
        -idiom_bonus."""
        out = []
        for L in self._idiom_lens:
            if p + L > len(keys):
                break
            for w, sylls in self.idioms.get(keys[p : p + L], ()):
                toks = self._han_tokens(w, sylls)
                if toks:
                    out.append((p + toks[0][1], toks[0][0], -self.idiom_bonus, tuple((t, p + e) for t, e in toks[1:])))
        ends = np.array([o[0] for o in out], dtype=np.int32)
        tids = np.array([o[1] for o in out], dtype=np.int32)
        costs = np.array([o[2] for o in out], dtype=np.float32)
        fixes_, rests = np.empty(len(out), dtype=object), np.empty(len(out), dtype=object)
        fixes_[:] = [()] * len(out)
        rests[:] = [o[3] for o in out]
        return ends, tids, costs, fixes_, rests

    def _han_tokens(self, w, sylls):
        """((token id, key end offset), ...) of the Han text `w` read as `sylls`, () when its tokens are not whole
        characters."""
        toks = self._idiom_tokens.get(w)
        if toks is None:
            toks, at, done = [], 0, 0
            for t in self.tok.encode(w):
                piece = self.tok.decode([t])
                done += len(piece)
                at = sum(len(x) for x in sylls[:done])
                toks.append((t, at))
                self.tok_text.setdefault(t, piece)
            ok = done == len(w) and all(self.tok_text[t] and all("一" <= c <= "鿿" for c in self.tok_text[t])
                                        for t, _ in toks)
            toks = self._idiom_tokens[w] = tuple(toks) if ok else ()
        return toks

    def _whole_words(self, context, keys, out):
        """`out` with the words that the keys are the whole pinyin of (word_top) added if the search has not got them
        before place word_slot: scored with the first of `out` by the model after the context (score), each word
        goes to its rank there but at the latest place word_slot, its path with it."""
        at = min(self.word_slot, len(out))
        add = [(w, sylls) for w, sylls in self.words.get(keys, ()) if w not in out[:at] and self._han_tokens(w, sylls)]
        add = add[: self.word_n]
        if not add:
            return out
        names = [w for w, _ in add]
        for w, sylls in add:
            self.paths.setdefault(w, tuple(self._han_tokens(w, sylls)))
        rest = [t for t in out if t not in names]
        head = rest[:at]
        scores = self.score(context, keys, head + names)
        for w, sc in zip(names, scores[len(head) :]):
            place = sum(x > sc for x in scores[: len(head)])
            head.insert(min(place, at), w)
        return head + rest[at:]

    def whole_pinyin(self, w):
        """`w` splits into whole syllables of pinyin."""
        ok = [True] + [False] * len(w)
        for i in range(1, len(w) + 1):
            ok[i] = any(ok[i - L] and w[i - L : i] in self.pinyin for L in range(1, min(6, i) + 1))
        return ok[len(w)]

    def feasible(self, keys, swaps=True):  # alts need nothing here: en etc. are readings of other characters
        """feasible[p]: keys[p:] can be split into reading prefixes, or (with swap_penalty) whole syllables
        with two adjacent keys exchanged, or (typo_penalty) typed with one key off (else a hypothesis at p is a dead end)."""
        n = len(keys)
        jumps = [[] for _ in range(n)]  # p -> ends of spans readable with one swap or typo
        if swaps and self.typos:
            for s in range(n):
                jumps[s] += [s + m for m in range(2, min(8, n - s + 1)) if keys[s : s + m] in self.typos]
        if swaps and self.swap_penalty:
            for j in range(n - 1):
                if keys[j] == keys[j + 1]:
                    continue
                ks = keys[:j] + keys[j + 1] + keys[j] + keys[j + 2:]
                for s in range(max(0, j - 5), j + 1):
                    for e in range(j + 1, min(s + 6, n) + 1):
                        if ks[s:e] not in self.syllables:
                            continue
                        if e > j + 1:
                            jumps[s].append(e)
                        else:  # the swap straddles into the next syllable, which must be whole too
                            jumps[s] += [e2 for e2 in range(e + 1, min(e + 6, n) + 1) if ks[e:e2] in self.syllables]
        if self.english_penalty:
            for s in range(n):
                jumps[s] += [s + L for L in range(2, min(20, n - s) + 1) if keys[s : s + L] in self.english]
        ok = [False] * n + [True]
        for p in range(n - 1, -1, -1):
            ok[p] = any(ok[p + L] for L in range(1, min(6, n - p) + 1) if keys[p : p + L] in self.prefixes) or any(
                ok[e] for e in jumps[p])
        return ok

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
        With swap_penalty, keys may also be read with two adjacent letters exchanged, and characters by EXTRA_READINGS
        (see edges); for a fine-tuned model a good enough such reading costs a second search with the keys it learned
        (swapped, or n for en) in the prompt. With single_weight, keys that are one whole syllable put its common
        characters first (_single_first).
        """
        self.paths = {}
        if not keys:
            return []
        single = self.single_weight and keys in self.syllables and any(v in keys for v in "aeiouv")
        if single and k <= SINGLE_PAGE:
            # the first k are its characters whatever the search finds: no search (5B.6: wo, hen 40-80 ms -> 2-4 on M4)
            page = self._single_first(context, keys, mix, [])
            if len(page) >= k and all(_gb2312(t) for t in page[:k]):  # else page is short: the search comes next
                return page[:k]
            self.paths = {}
        found = self._search(context, keys, k, beam, mix, prefix, stop, eos, fixes=True)
        if found and self.keys_in_prompt:
            # a fixed reading of a text the typed keys also read as typed stays, and is no reason to search again
            found = [f if not f[3] or not self._reads(f[1], keys) else (*f[:3], ()) for f in found]
            best, todo = found[0][0], []
            for score, _t, _path, fix in found:  # judged with alt_bonus, without swap_penalty
                if fix and score + sum(x[3] for x in fix if x[3] > 0) >= best - self.swap_margin and fix not in todo \
                        and len(todo) < self.swap_retries:
                    todo.append(fix)
            if todo:
                found = [f for f in found if not f[3]]
                for fix in todo:
                    swaps = [x for x in fix if x[3] > 0]
                    swapped, cost = _apply(keys, swaps), sum(x[3] for x in swaps)
                    alt = len(swaps) < len(fix)
                    pre = prefix if prefix and prefix[-1][1] <= min(x[0] for x in fix) else None  # keys left alone
                    # an extra reading (en for 嗯): the dictionary's (n) in the prompt, while the search still reads
                    # the keys as typed (swaps made) and keeps only what reads one; alt_bonus is in its scores
                    again = self._search(context, swapped, k, beam, mix, pre, stop, eos, fixes=False, alts=alt,
                                         prompt_keys=_apply(keys, fix) if alt else None)
                    if again is None:
                        return None
                    found += [(score - cost, t, _typed_path(path, swaps), fix) for score, t, path, f in again if f or not alt]
                    self.stats["swap_retries"] = self.stats.get("swap_retries", 0) + 1
                found.sort(key=lambda f: f[:3], reverse=True)
        if found is None:
            return None
        out = []
        for _s, t, path, _fix in found:
            if t not in self.paths:
                self.paths[t] = path
                out.append(t)
        out = _simplified_first(out)  # 這樣 after 这样 / 着养 / 遮阳 / 蒸养 (5B.7)
        if single:
            out = self._single_first(context, keys, mix, out)
        if self.word_top and k > self.word_slot:
            out = self._whole_words(context, keys, out)
        if keys in self.english and not self._full_pinyin_so_far(keys):
            out = [keys] + [t for t in out if t != keys]  # 5B.9
        return out[:k]

    def _full_pinyin_so_far(self, keys):
        """Whether keys are whole pinyin syllables, the last one maybe not typed to its end (long, bilibili, bus of
        bushi, yes of yeshi; not github, cpu, cmd, follow)."""
        ok = [True] + [False] * len(keys)
        for i in range(1, len(keys) + 1):
            ok[i] = any(ok[i - L] and keys[i - L : i] in self.pinyin for L in range(1, min(6, i) + 1))
        n = len(keys)
        return any(ok[i] and keys[i:] in self.pinyin_starts for i in range(max(0, n - 5), n)) or ok[n]

    def _single_first(self, context, keys, mix, out):
        """For keys that are one whole syllable: its characters ranked by log frequency + single_weight x their
        score at the first step (as in segments) + alt_bonus for an extra reading (en: 嗯 before 恩) first,
        SINGLE_PAGE of them and only GB2312 ones; then the search's other candidates that read the keys as whole syllables (西安 for xian), the rest of those characters, and
        last what abbreviates a syllable (河南 for hen as he + n, 真爱 for zai). Without the n / ng / m of 嗯 呣:
        n is mostly the start of ni."""
        n = len(keys)
        plain = context[-64:] if context.strip() else self.empty_prompt
        self._prefill(fmt.prompt(context, keys, self.order) if self.keys_in_prompt else plain)
        mix_w = 0
        if self.mix and mix:
            self.mix.prefill(self.tok.encode(plain))
            mix_w = self.mix_weight if context.strip() else self.mix_weight_empty
        ends, tids, abs_, fixes = self.edges(keys, 0, swaps=False)
        ok = np.array([e == n and len(self.tok_text[int(t)]) == 1 and abs(a - sum(x[3] for x in f)) < 1e-6
                       for e, t, a, f in zip(ends, tids, abs_, fixes)], dtype=bool)
        if not ok.any():
            return out
        tids, bonus = tids[ok], [-sum(x[3] for x in f) for f in fixes[ok]]
        cols = np.unique(tids)
        lp = np.array(_log_softmax_cols(self._ctx_logits, mx.array(cols)))[0]
        if mix_w:
            lp = lp + mix_w * np.array(_log_softmax_cols(self.mix.ctx_logits, mx.array(cols)))[0]
        floor = min(self.char_logp.values())
        # alt_bonus whole, not scaled: it is how the user types (嗯 as en), not the model's opinion
        score = [self.single_weight * lp[np.searchsorted(cols, t)] + b + self.char_logp.get(self.tok_text[int(t)], floor)
                 for t, b in zip(tids, bonus)]
        singles = []
        for j in np.argsort(score, kind="stable")[::-1]:
            t = self.tok_text[int(tids[j])]
            if t not in singles:
                singles.append(t)
                self.paths.setdefault(t, ((int(tids[j]), n),))
        page = [t for t in singles if _gb2312(t)][:SINGLE_PAGE]  # not 妳 載 楽 (traditional, variants) up front
        whole = [t for t in out if t not in page and self._reads(t, keys, extra=True, whole=True)]
        rest = [t for t in singles if t not in page and t not in whole]
        return page + whole + rest + [t for t in out if t not in page and t not in whole and t not in rest]

    def _reads(self, text, keys, extra=False, whole=False):
        """Whether `keys` read as `text` with no swap: each character a non-empty prefix of one of its readings
        (with `extra`, EXTRA_READINGS' too; with `whole`, the whole reading)."""
        table = self.char_readings_extra if extra else self.char_readings
        ends = {0}
        for ch in text:
            rs = table.get(ch, ())
            ends = {e + L for e in ends for r in rs for L in range(len(r) if whole else 1, len(r) + 1)
                    if keys[e : e + L] == r[:L]}
            if not ends:
                return False
        return len(keys) in ends

    def _search(self, context, keys, k, beam, mix, prefix, stop, eos, fixes, alts=None, prompt_keys=None):
        """The beam search of convert: finished hypotheses (score, text, path, fixes to the keys) best first,
        or None when `stop` dropped it. `fixes`: readings that need fixes too (`alts`: extra readings only).
        The keys in the prompt are `prompt_keys`, by default `keys`."""
        alts = fixes if alts is None else alts
        beam = beam or self.beam
        mix = bool(self.mix) and mix
        plain = context[-64:] if context.strip() else self.empty_prompt
        if self.keys_in_prompt:
            self._prefill(fmt.prompt(context, prompt_keys or keys, self.order))
        else:
            self._prefill(plain)
        if mix:
            self.mix.prefill(self.tok.encode(plain))
        n = len(keys)
        feasible = self.feasible(keys, fixes)
        if not feasible[0]:
            return []
        edge_cache = {}

        def edges_at(p):
            if p not in edge_cache:
                e = self.edges(keys, p, swaps=fixes, alts=alts)
                rests = np.empty(len(e[0]), dtype=object)
                rests[:] = [()] * len(e[0])
                e = (*e, rests)
                if self.english_penalty:
                    e = tuple(np.concatenate([a, b]) for a, b in zip(e, self.english_edges(keys, p)))
                if self.idiom_bonus:
                    e = tuple(np.concatenate([a, b]) for a, b in zip(e, self.idiom_edges(keys, p)))
                last = np.array([r[-1][1] if r else x for x, r in zip(e[0], e[4])], dtype=np.int32)
                ok = np.array([feasible[x] for x in last], dtype=bool)
                edge_cache[p] = tuple(x[ok] for x in e) if len(e[0]) else e
            return edge_cache[p]

        logits = self._ctx_logits
        mix_logits = self.mix.ctx_logits if mix else None
        mix_w = (self.mix_weight if context.strip() else self.mix_weight_empty) if mix else 0
        active = [(0.0, "", 0, (), (), ())]  # (score, text, key position, path, fixes, rest of an English word)
        cache = mix_cache = None
        mix_rows = [0] if mix_w else []  # the rows of active that mix_logits has, in its order
        if prefix:  # not into an English word: it may be cut short there (pyth of python)
            prefix = list(prefix)
            while prefix and prefix[-1][0] in self.tok_text and self.tok_text[prefix[-1][0]].isascii():
                prefix.pop()
        if prefix and prefix[-1][1] <= n and feasible[prefix[-1][1]]:
            path = tuple(prefix)
            cache = self._fork_cache(mx.array([0]), len(path) + n + 2)
            ptoks = mx.array([t for t, _e in path])[None]
            logits = self.model(ptoks, cache=cache)[:, -1, :]
            if mix_w:
                mix_cache = self.mix.fork(mx.array([0]), len(path) + n + 2)
                mix_logits = self.mix.model(ptoks, cache=mix_cache)[:, -1, :]
            active = [(0.0, "".join(self.tok_text[t] for t, _e in path), path[-1][1], path, (), ())]
        finished = []
        for _step in range(n + 1):
            if stop and stop():
                return None
            # Only the columns this step can use (the tokens that fit at the beams' key positions, and EOS)
            # leave the GPU: copying the whole vocab per beam cost more than the forward.
            pos = {h[2] for h in active if h[2] < n and not h[5]}
            more = [np.array([h[5][0][0] for h in active if h[5]], np.int32)]
            cols = np.unique(np.concatenate([edges_at(p)[1] for p in pos] + more + [np.array([self.eos], np.int32)]))
            col_of = lambda t: np.searchsorted(cols, t)  # noqa: E731
            lp = _log_softmax_cols(logits, mx.array(cols))
            if mix_w and mix_rows:
                mlp = _log_softmax_cols(mix_logits, mx.array(cols))
                mx.eval(lp, mlp)
                lp, mlp = np.array(lp), np.array(mlp)
                eos_lp = lp[:, col_of(self.eos)].copy()
                lp[mix_rows] += mix_w * mlp
                lp[:, col_of(self.eos)] = eos_lp
            else:
                lp = np.array(lp)
            cand_s, cand_parent, cand_tid, cand_end, cand_sw, cand_rest = [], [], [], [], [], []
            for b, (score, text, p, path, fix, rest) in enumerate(active):
                if rest:  # inside an English word: its next token only
                    ends, tids, abs_ = (np.array([x], np.int32) for x in (rest[0][1], rest[0][0], 0))
                    fixes_, rests = np.empty(1, dtype=object), np.empty(1, dtype=object)
                    fixes_[0], rests[0] = (), rest[1:]
                elif p == n:  # all keys consumed, waiting for its end-of-sentence score
                    finished.append((score + float(lp[b, col_of(self.eos)]), text, path, fix))
                    continue
                else:
                    ends, tids, abs_, fixes_, rests = edges_at(p)
                if len(tids) == 0:
                    continue
                cand_s.append(score + lp[b, col_of(tids)] - abs_)
                cand_parent.append(np.full(len(tids), b))
                cand_tid.append(tids)
                cand_end.append(ends)
                cand_sw.append(fixes_)
                cand_rest.append(rests)
            if not cand_s:
                break
            S, P, T, E, W, R = (np.concatenate(x) for x in (cand_s, cand_parent, cand_tid, cand_end, cand_sw, cand_rest))
            order = np.argsort(-(S + self.beta * E))

            def grown(j):  # (score, text, path, fixes) of candidate j; only built for the ones kept
                h = active[P[j]]
                return float(S[j]), h[1] + self.tok_text[int(T[j])], h[3] + ((int(T[j]), int(E[j])),), h[4] + W[j]

            finish_now = not (self.keys_in_prompt and eos)  # else one more step scores the end of the sentence
            nxt, parents, tids = [], [], []
            typo_rows = 0
            for i, j in enumerate(order):
                if E[j] == n and finish_now and not R[j]:
                    finished.append(grown(j))
                elif len(nxt) - typo_rows < beam:
                    s, text, path, fix = grown(j)
                    if fix and any(sorted(keys[a:b]) != sorted(new) for a, b, new, _c in fix):
                        # typo readings get a few rows of their own, beside the beam of the keys as typed (5B.8)
                        if typo_rows >= self.typo_rows:
                            continue
                        typo_rows += 1
                    nxt.append((s, text, int(E[j]), path, fix, R[j]))
                    parents.append(int(P[j]))
                    tids.append(int(T[j]))
                if len(nxt) - typo_rows >= beam:  # the beam is full: only finished ones still count, up to k * 3 of them
                    if finish_now and len(finished) < k * 3:
                        rest = order[i + 1 :]
                        finished += [grown(j) for j in rest[E[rest] == n] if not R[j]][: k * 3 - len(finished)]
                    break
            finished.sort(key=lambda f: f[:3], reverse=True)
            if not nxt or (len(finished) >= k and finished[k - 1][0] >= max(h[0] for h in nxt)):
                break
            texts = list(dict.fromkeys(f[1] for f in finished))
            if len(texts) >= k:
                # a beam waiting for its end-of-sentence score, already below the k-th text finished, stays below it:
                # no forward for it (5B.6: about an eighth of the rows). Not one with a fix, which may still be
                # searched again (convert).
                floor = next(f[0] for f in finished if f[1] == texts[k - 1])
                keep = [i for i, h in enumerate(nxt) if h[2] < n or h[4] or h[5] or h[0] > floor]
                if len(keep) < len(nxt):
                    nxt, parents, tids = [nxt[i] for i in keep], [parents[i] for i in keep], [tids[i] for i in keep]
            idx = mx.array(parents)
            if cache is None:
                cache = self._fork_cache(idx, n + 2)
            else:
                self._reorder(cache, idx)
            t0 = time.perf_counter()
            logits = self.model(mx.array(tids)[:, None], cache=cache)[:, -1, :]
            # the mixed-in model only for the beams with keys left: the others wait for the end-of-sentence score,
            # which is the first model's alone
            rows = [i for i, h in enumerate(nxt) if h[2] < n or h[5]] if mix_w else []
            if rows:
                at = {a: i for i, a in enumerate(mix_rows)}
                midx = mx.array([at[parents[i]] for i in rows])
                mix_cache = self.mix.fork(midx, n + 2) if mix_cache is None else (self._reorder(mix_cache, midx) or mix_cache)
                mix_logits = self.mix.model(mx.array([tids[i] for i in rows])[:, None], cache=mix_cache)[:, -1, :]
                mx.eval(logits, mix_logits)
            else:
                mx.eval(logits)
            mix_rows = rows
            self.stats["forward_ms"] += (time.perf_counter() - t0) * 1000
            self.stats["calls"] += 1
            active = nxt
        finished.sort(key=lambda f: f[:3], reverse=True)
        return finished

    def segments(self, context, keys, words=30):
        """Candidates for the start of `keys` only, for picking a sentence piece by piece (4A.10): [(text, keys used)],
        words of two or more characters first (at most `words`), then every single character of the first syllable;
        those with a traditional-only character last.
        Each is one token starting at key 0, scored like convert's first step (same prompt, so its cache is reused;
        one log-softmax, no forward), the rest of the keys left readable. Readings that abbreviate a syllable count
        only when nothing reads whole syllables (wmxz: 我们; xian: not 仙女 as xia + n; de: not 第二). Words longer than one
        token, and characters the tokenizer has no token for, are not among them.
        """
        n = len(keys)
        if not n:
            return []
        plain = context[-64:] if context.strip() else self.empty_prompt
        self._prefill(fmt.prompt(context, keys, self.order) if self.keys_in_prompt else plain)
        mix_w = 0
        if self.mix:
            self.mix.prefill(self.tok.encode(plain))
            mix_w = self.mix_weight if context.strip() else self.mix_weight_empty
        feasible = self.feasible(keys, swaps=False)  # as in convert, but no swaps: pieces are read as typed
        ends, tids, abs_, fixes = self.edges(keys, 0, swaps=False)  # extra readings (嗯 as en) count, as typed
        ok = np.array([feasible[e] for e in ends], dtype=bool)
        if not ok.any():
            return []
        ends, tids, abs_, fixes = ends[ok], tids[ok], abs_[ok], fixes[ok]
        cols = np.unique(tids)
        lp = np.array(_log_softmax_cols(self._ctx_logits, mx.array(cols)))[0]
        if mix_w:
            lp = lp + mix_w * np.array(_log_softmax_cols(self.mix.ctx_logits, mx.array(cols)))[0]
        score = lp[np.searchsorted(cols, tids)] - abs_
        groups = {(w, whole): [] for w in (True, False) for whole in (True, False)}  # (word?, whole syllables?)
        seen = set()
        for j in np.argsort(-score, kind="stable"):
            t = self.tok_text[int(tids[j])]
            if t in seen:  # the same text read with fewer keys scores lower: keep the best reading
                continue
            seen.add(t)
            whole = abs(abs_[j] - sum(x[3] for x in fixes[j])) < 1e-6  # no abbreviated char (alt_bonus aside)
            groups[(len(t) >= 2, whole)].append((t, int(ends[j])))
        whole = bool(groups[(True, True)] or groups[(False, True)])
        return _simplified_first(groups[(True, whole)][:words] + groups[(False, whole)], lambda s: s[0])  # 這樣 說 last (5B.7)

    def score(self, context, keys, texts):
        """The model's score of each of `texts` as the reading of `keys` after `context`, as the search would add it
        up (same prompt, the mixed-in model with its weight, the end of the sentence for a fine-tuned model) but with
        the tokenizer's own tokens and no abbreviation penalties: for comparing texts with each other (5A.3's
        user words against the search's best), not with the search's scores. One forward per model for all."""
        if not texts:
            return []
        ids = [self.tok.encode(t) for t in texts]
        L = max(len(x) for x in ids)
        batch = mx.array([x + [self.eos] * (L - len(x)) for x in ids])
        idx = mx.array([0] * len(texts))
        plain = context[-64:] if context.strip() else self.empty_prompt
        self._prefill(fmt.prompt(context, keys, self.order) if self.keys_in_prompt else plain)
        runs = [(self.model, self._fork_cache(idx, L + 1), self._ctx_logits, 1.0)]
        if self.mix:
            self.mix.prefill(self.tok.encode(plain))
            mix_w = self.mix_weight if context.strip() else self.mix_weight_empty
            runs.append((self.mix.model, self.mix.fork(idx, L + 1), self.mix.ctx_logits, mix_w))
        target = mx.concatenate([batch, mx.full((len(texts), 1), self.eos, dtype=batch.dtype)], axis=1)
        out = np.zeros(len(texts))
        for i, (model, cache, ctx_logits, w) in enumerate(runs):
            logits = mx.concatenate([mx.repeat(ctx_logits[:, None, :], len(texts), axis=0), model(batch, cache=cache)],
                                    axis=1)  # position j predicts token j; position len(text) its end
            lp = mx.take_along_axis(logits, target[..., None], axis=-1)[..., 0] - mx.logsumexp(logits, axis=-1)
            lp = np.array(lp.astype(mx.float32))  # only the targets' columns leave the GPU
            for b, x in enumerate(ids):
                out[b] += w * float(lp[b, : len(x)].sum())
                if i == 0 and self.keys_in_prompt:
                    out[b] += float(lp[b, len(x)])
        return out.tolist()


def _apply(keys, fix):
    """keys with the fixes (start, end, standard keys, cost) made."""
    out, i = [], 0
    for s, e, new, _c in sorted(fix):
        out += keys[i:s] + new
        i = e
    return "".join(out) + keys[i:]


def _typed_path(path, fix):
    """A path [(token, key end)] searched on _apply(keys, fix) with its key ends moved back onto keys (a typo fix
    can change the length)."""
    fix = sorted(fix)
    if all(e - s == len(new) for s, e, new, _c in fix):
        return path

    def back(p):
        d = 0
        for s, e, new, _c in fix:
            if p <= s + d:
                break
            if p < s + d + len(new):
                return e if p - d > s else s  # inside the fixed keys: to its nearer end
            d += len(new) - (e - s)
        return p - d

    return tuple((t, back(p)) for t, p in path)


ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
KEY_AT = {c: (y, x + (0, 0.25, 0.75)[y]) for y, row in enumerate(ROWS) for x, c in enumerate(row)}


def adjacent(a, b):
    """Keys next to each other on a QWERTY keyboard (diagonals too)."""
    (ya, xa), (yb, xb) = KEY_AT[a], KEY_AT[b]
    return a != b and (abs(xa - xb) == 1 if ya == yb else abs(ya - yb) == 1 and abs(xa - xb) <= 0.75)


def _typos(syllables, prefixes, penalty):
    """{typed keys: ((syllable, cost), ...)}: each syllable typed with one key off, as the typing log shows them (5B.8):
    one key too many inside it or its last vowel twice (penalty, +1 when not next to or the same as a neighbour),
    one missing, or one next to the right one, not its first key (a different first key is mostly another word).
    Only keys that cannot be read as typed at all (not a reading prefix, so never an abbreviation or a syllable
    typed halfway), with a vowel, two keys or more."""
    out = {}

    def add(typed, sy, cost):
        if typed[:-1] == sy and typed[-1] not in "aeiouv":
            return  # a consonant after the syllable starts the next character (duann: 断你)
        if len(typed) >= 2 and typed not in prefixes and any(v in typed for v in "aeiouv"):
            if out.setdefault(typed, {}).get(sy, 99) > cost:
                out[typed][sy] = cost

    letters = "abcdefghijklmnopqrstuvwxyz"
    for sy in syllables:
        for i in range(1, len(sy)):
            for c in letters:
                near = c in (sy[i - 1], sy[i]) or adjacent(c, sy[i - 1]) or adjacent(c, sy[i])
                add(sy[:i] + c + sy[i:], sy, penalty if near else penalty + 1)
            add(sy[:i] + sy[i + 1 :], sy, penalty)
            for c in letters:
                if adjacent(c, sy[i]):
                    add(sy[:i] + c + sy[i + 1 :], sy, penalty)
        if sy[-1] in "aeiouv":
            add(sy + sy[-1], sy, penalty)  # yii
    return {t: tuple(sorted(d.items(), key=lambda x: x[1])) for t, d in out.items()}


_TRAD = None


def _simplified_first(texts, text=lambda t: t):
    """texts with those whose text(t) has a traditional-only character (ime/trad_chars.txt: 這 樣, not 乾 於 啰)
    moved to the end, order kept otherwise."""
    global _TRAD
    if _TRAD is None:
        _TRAD = frozenset("".join(line.strip() for line in open(TRAD_CHARS, encoding="utf-8") if not line.startswith("#")))
    trad = [any(c in _TRAD for c in text(t)) for t in texts]
    return [t for t, x in zip(texts, trad) if not x] + [t for t, x in zip(texts, trad) if x]


def _english_words():
    return frozenset(line.strip() for line in open(EN_WORDS, encoding="utf-8") if not line.startswith("#"))


def _idioms():
    """({whole pinyin: [(idiom, syllables)]}, the lengths of those keys sorted)."""
    table = {}
    for line in open(IDIOMS, encoding="utf-8"):
        if not line.startswith("#"):
            w, py = line.rstrip("\n").split("\t")
            sylls = tuple(py.split())
            table.setdefault("".join(sylls), []).append((w, sylls))
    return table, tuple(sorted({len(k) for k in table}))


def _words(top):
    """{whole pinyin: [(word, syllables)]} of the `top` most frequent words of ime/words.tsv (most frequent first)."""
    table = {}
    for line in open(WORDS, encoding="utf-8"):
        if line.startswith("#"):
            continue
        if top <= 0:
            break
        w, py, _ = line.rstrip("\n").split("\t")
        sylls = tuple(py.split())
        table.setdefault("".join(sylls), []).append((w, sylls))
        top -= 1
    return table


def _gb2312(text):
    try:
        text.encode("gb2312")
        return True
    except UnicodeEncodeError:
        return False


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
