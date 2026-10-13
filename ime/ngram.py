"""Baseline converter: word lexicon + word bigram + Viterbi beam, like a classic pinyin IME.

Matching rule shared with the LLM engine: every character consumes a non-empty prefix of one of its
readings, so the same lattice covers full pinyin, initials, and half-typed syllables.
"""

import itertools
import math
import pickle
import re
import unicodedata

import jieba
from pypinyin.pinyin_dict import pinyin_dict

from ime.pinyin import syllables

jieba.setLogLevel(60)
HAN_RUN = re.compile(r"[一-鿿]+")
BOS = "<s>"


def toneless(r: str) -> str:
    r = unicodedata.normalize("NFC", r)
    for u in "üǖǘǚǜ":
        r = r.replace(u, "v")
    return "".join(c for c in unicodedata.normalize("NFD", r) if not unicodedata.combining(c))


# Readings people type that the dictionary lacks: character -> {typed reading: the dictionary's reading, which the
# fine-tuned models learned it by}. From the user's WeChat text: 嗯 (216 times, 194 of them 嗯呢) is typed en.
EXTRA_READINGS = {"嗯": {"en": "n"}}


def char_readings_table(extra: bool = False) -> dict[str, list[str]]:
    table = {}
    for cp, rs in pinyin_dict.items():
        ch = chr(cp)
        if "一" <= ch <= "鿿":
            table[ch] = sorted({toneless(r) for r in rs.split(",")})
    if extra:
        for ch, alts in EXTRA_READINGS.items():
            table[ch] = sorted(set(table.get(ch, ())) | set(alts))
    return table


def with_extra(text: str, sylls: list[str]) -> list[list[str]]:
    """sylls (one per character of text), and every variant with EXTRA_READINGS' typed readings in its place."""
    choices = [[sy] + [a for a, std in EXTRA_READINGS.get(ch, {}).items() if std == sy] for ch, sy in zip(text, sylls)]
    return [list(v) for v in itertools.product(*choices)]


class Node:
    __slots__ = ("kids", "by_letter", "words", "alt")

    def __init__(self):
        self.kids: dict[str, Node] = {}
        self.by_letter: dict[str, list[str]] = {}
        self.words: list[str] = []
        self.alt: dict = {}  # word -> ((syllable index, the dictionary's reading, the typed one), ...) when only EXTRA_READINGS reach it here


class NgramEngine:
    name = "ngram"

    def __init__(self, path="data/corpus/ngram.pkl", beam=24, abbrev_penalty=1.0, discount=0.75):
        d = pickle.load(open(path, "rb"))
        self.uni: dict[str, int] = d["unigrams"]
        self.bi: dict[tuple[str, str], int] = d["bigrams"]
        self.total = sum(self.uni.values())
        self.beam, self.alpha, self.D = beam, abbrev_penalty, discount
        self.uni[BOS] = d["bos_count"]
        # Per history word: total count as a history and number of distinct followers (for backoff weight).
        self.hist: dict[str, list[int]] = {}
        for (a, _), c in self.bi.items():
            h = self.hist.setdefault(a, [0, 0])
            h[0] += c
            h[1] += 1
        self.root = Node()
        chars = char_readings_table(extra=True)
        for ch, rs in chars.items():
            for r in rs:
                self._insert([r], ch)
            self.uni.setdefault(ch, 0)
        for w in self.uni:
            if len(w) > 1 and w != BOS:
                for sylls in with_extra(w, syllables(w)):
                    self._insert(sylls, w)
        self._finish(self.root)

    def _insert(self, sylls, word):
        n = self.root
        for s in sylls:
            n = n.kids.setdefault(s, Node())
        n.words.append(word)

    def _finish(self, n):
        for s, k in n.kids.items():
            n.by_letter.setdefault(s[0], []).append(s)
            self._finish(k)

    # --- language model -------------------------------------------------------
    def p_uni(self, w):
        return (self.uni.get(w, 0) + 0.5) / (self.total + 0.5 * len(self.uni))

    def logp(self, prev, w):
        h = self.hist.get(prev)
        if not h:
            return math.log(self.p_uni(w))
        c = self.bi.get((prev, w), 0)
        p = max(c - self.D, 0) / h[0] + self.D * h[1] / h[0] * self.p_uni(w)
        return math.log(p)

    # --- lattice --------------------------------------------------------------
    def edges(self, keys, p):
        """All (end, word, n_abbreviated_chars) for words starting at key position p."""
        out = []

        def walk(node, q, abbrev):
            if node is not self.root:
                for w in node.words:
                    out.append((q, w, abbrev))
            if q >= len(keys):
                return
            for s in node.by_letter.get(keys[q], ()):
                kid = node.kids[s]
                m = 1
                while m < len(s) and q + m < len(keys) and keys[q + m] == s[m]:
                    m += 1
                for L in range(1, m + 1):
                    walk(kid, q + L, abbrev + (L < len(s)))

        walk(self.root, p, 0)
        return out

    def convert(self, context, keys, k=5):
        runs = HAN_RUN.findall(context[-32:])
        prev0 = BOS
        if runs and context.rstrip()[-1:] and HAN_RUN.match(context.rstrip()[-1]):
            prev0 = jieba.lcut(runs[-1], HMM=False)[-1]
            if prev0 not in self.hist:
                prev0 = BOS
        n = len(keys)
        # beams[p]: list of (score, last_word, text)
        beams: list[list[tuple[float, str, str]]] = [[] for _ in range(n + 1)]
        beams[0] = [(0.0, prev0, "")]
        for p in range(n):
            if not beams[p]:
                continue
            beams[p] = self._prune(beams[p])
            es = self.edges(keys, p)
            for score, last, text in beams[p]:
                for end, w, ab in es:
                    beams[end].append((score + self.logp(last, w) - self.alpha * ab, w, text + w))
            for q in range(p + 1, n + 1):
                if len(beams[q]) > self.beam * 4:
                    beams[q] = self._prune(beams[q])
        best, seen = [], set()
        for s, _, t in sorted(self._prune(beams[n]), reverse=True):
            if t not in seen:
                seen.add(t)
                best.append(t)
        return best[:k]

    def _prune(self, hyps):
        best = {}
        for h in hyps:
            key = (h[1], h[2])
            if key not in best or best[key][0] < h[0]:
                best[key] = h
        return sorted(best.values(), reverse=True)[: self.beam]
