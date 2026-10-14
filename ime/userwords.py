"""The user's words (5A.3, N-009 layer 1): what the user picked over the engine's first candidate comes first next time.

A word list on this Mac only (default ~/Library/Application Support/llm-ime/words.json): keys -> {word: [count, last
used (unix time)]}. The input method tells the service what it committed (`learn`: the keys, the text, and the
engine's first candidate then); where the text differs from that first candidate, the differing run of characters,
widened to at least two, is learned with its keys (liji picked as 力积: liji -> 力积; 这些怎么得 for zhexiezenmede over
这些怎么的: mede -> 么得). A text that is the first candidate only counts again the learned words in it.

After each search, `rerank` looks for learned keys inside the typed keys, at character boundaries of the best
candidate; the best with that stretch replaced by the learned word (or the word itself, when it is all the keys) is
scored by the model against the best (LLMEngine.score) and gets `bonus`: when that puts it ahead it comes first, else
second. Keys with no learned keys in them cost nothing. `segments` puts learned words for the start of the keys in
front of the engine's (4A.10).
"""

import json
import math
import os
import threading
import time

from ime.ngram import char_readings_table

DEFAULT_PATH = os.path.expanduser("~/Library/Application Support/llm-ime/words.json")


def spans(text, keys, readings):
    """Each character's keys [(start, end)] when `keys` read as `text` (a character takes a non-empty prefix of one of
    its readings; whole readings preferred), or None."""
    n = len(keys)
    best = {0: (0, [])}  # key position -> (abbreviated characters, spans)
    for ch in text:
        nxt = {}
        for p, (cost, sp) in best.items():
            for r in readings.get(ch, ()):
                for L in range(len(r), 0, -1):
                    if keys[p : p + L] == r[:L]:
                        c = cost + (L < len(r))
                        if p + L not in nxt or c < nxt[p + L][0]:
                            nxt[p + L] = (c, sp + [(p, p + L)])
        if not nxt:
            return None
        best = nxt
    return best[n][1] if n in best else None


class UserWords:
    def __init__(self, engine, path=DEFAULT_PATH, base=10.0, per_count=2.0, cap=16.0, half_life_days=30.0):
        """bonus (nats) = min(cap, base + per_count x ln(1 + count)), halved every half_life_days since last used."""
        self.engine, self.path = engine, path
        self.base, self.per_count, self.cap, self.half_life = base, per_count, cap, half_life_days * 86400
        self.readings = char_readings_table()
        self.lock = threading.Lock()
        self.words = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.words = json.load(f)
            except (OSError, ValueError):
                self.words = {}

    # --- learning (any thread) ------------------------------------------------------------------------------------

    def learn(self, keys, text, first, now=None):
        """The user committed `text` for `keys` when the engine's first candidate was `first`."""
        now = now or time.time()
        with self.lock:
            if text == first:
                for k, w in self._found(keys, text):
                    self.words[k][w] = [self.words[k][w][0] + 1, now]
            else:
                for k, w in self._differing(keys, text, first):
                    c = self.words.setdefault(k, {}).get(w, [0, now])[0]
                    self.words[k][w] = [c + 1, now]
            self._save()

    def add(self, keys, word, count=1, now=None):
        """A word from elsewhere (an import): count is how often it counts as picked."""
        with self.lock:
            c = self.words.setdefault(keys, {}).get(word, [0, 0])[0]
            self.words[keys][word] = [c + count, now or time.time()]

    def import_file(self, path, count=1):
        """Lines `keys<TAB>word[<TAB>count]`; returns how many were added."""
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2 or not parts[0].isalpha() or not parts[0].islower() or not parts[1]:
                    continue
                self.add(parts[0], parts[1], int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else count)
                n += 1
        with self.lock:
            self._save()
        return n

    def forget(self):
        with self.lock:
            self.words = {}
            if self.path and os.path.exists(self.path):
                os.remove(self.path)

    def _differing(self, keys, text, first):
        st, sf = spans(text, keys, self.readings), spans(first, keys, self.readings)
        if st is None:
            return []
        if sf is None:
            return [(keys, text)] if len(text) <= 4 else []
        same = {(s, ch) for s, ch in zip(sf, first)}
        diff = [(s, ch) not in same for s, ch in zip(st, text)]
        out, i = [], 0
        while i < len(text):
            if not diff[i]:
                i += 1
                continue
            j = i
            while j < len(text) and diff[j]:
                j += 1
            a, b = i, j
            if b - a < 2:  # one character alone says little: with its neighbour
                b, a = (b + 1, a) if b < len(text) else (b, max(0, a - 1))
            out.append((keys[st[a][0] : st[b - 1][1]], text[a:b]))
            i = j
        return out

    def _found(self, keys, text):
        """Learned (keys, word) that `text` contains where `keys` read it."""
        st = spans(text, keys, self.readings)
        if st is None:
            return []
        out = []
        for k, ws in self.words.items():
            for w in ws:
                for a in range(len(text) - len(w) + 1):
                    if text[a : a + len(w)] == w and keys[st[a][0] : st[a + len(w) - 1][1]] == k:
                        out.append((k, w))
                        break
        return out

    def _save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.words, f, ensure_ascii=False)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    # --- using them (the conversion thread) -----------------------------------------------------------------------

    def bonus(self, count, last, now=None):
        age = max(0.0, (now or time.time()) - last)
        return min(self.cap, self.base + self.per_count * math.log1p(count)) * 0.5 ** (age / self.half_life)

    def _variants(self, keys, best):
        """{variant text: bonus} for learned words whose keys are in `keys`."""
        with self.lock:
            hits = [(k, w, c, t) for k, ws in self.words.items() if k in keys for w, (c, t) in ws.items()]
        if not hits:
            return {}
        sp = spans(best, keys, self.readings) if best else None
        starts = {s[0]: i for i, s in enumerate(sp)} if sp else {}
        ends = {s[1]: i + 1 for i, s in enumerate(sp)} if sp else {}
        out = {}
        for k, w, c, t in hits:
            p = keys.find(k)
            while p >= 0:
                if p == 0 and len(k) == len(keys):
                    v = w
                elif p in starts and p + len(k) in ends:
                    v = best[: starts[p]] + w + best[ends[p + len(k)] :]
                else:
                    v = None
                if v and v != best:
                    out[v] = max(out.get(v, 0.0), self.bonus(c, t))
                p = keys.find(k, p + 1)
        return out

    def rerank(self, context, keys, cands):
        if not cands:
            return cands
        variants = self._variants(keys, cands[0])
        if not variants:
            return cands
        texts = [cands[0]] + list(variants)
        scores = self.engine.score(context, keys, texts)
        best = scores[0]
        ranked = sorted(((s + variants[v], v) for v, s in zip(texts[1:], scores[1:])), reverse=True)
        ahead = [v for s, v in ranked if s >= best]
        after = [v for s, v in ranked if s < best]
        head = ahead + [cands[0]] + after
        return head + [c for c in cands[1:] if c not in head]

    def segments(self, keys, segs):
        """Learned words for the start of `keys` (not all of them) in front of the engine's pieces [(text, keys used)]."""
        with self.lock:
            mine = sorted(((self.bonus(c, t), w, len(k)) for k, ws in self.words.items()
                           if len(k) < len(keys) and keys.startswith(k) for w, (c, t) in ws.items()), reverse=True)
        front = [(w, n) for _b, w, n in mine]
        return front + [s for s in segs if s not in front]
