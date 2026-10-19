"""Common words typed after a context (6B.5: 还有只修 + chengyu gave 成与 成羽 成鱼 first, 成语 10th). Engine variants
side by side, the engine itself unchanged: they are patched in here.

  .venv/bin/python scripts/word_ctx_eval.py --model models/r6-4bit [--n 300] [--variants base,tok2,word3] [--sentences]

Items, public only: LCCC test utterances and held-out Wikipedia lines cut into words by jieba; one word per line of two
to four Han characters among jieba's 20k most frequent, not the line's first word, with at least two Han characters
before it in the line; the context is the text before it (64 characters at most), the keys its full pinyin (also as
initials with --initials). Two sets: "word", any such word; "cross", only words whose first character makes a
two-character jieba word with the context's last (修 + 成: 修成). Prints P@1 / P@5 per variant, the first items the
first variant misses, and the user's case.
--sentences: also P@1 / P@5 of the public sentences of data/private/eval.jsonl (full and initials), so a variant that
helps words but hurts sentences shows.

Variants (name + number of nats):
  base     the v02 engine as it is
  tokN     N nats per token read: fewer, longer tokens (which are mostly words) first
  wordN    N nats off each token of two or more Han characters that is one of jieba's 20k most frequent words
  firstN   N nats for a first token of one Han character that makes a jieba word with the context's last character
           (修 + 成: 修成), the context taken to end where a word does
  endN     the search's 10 best rescored (LLMEngine.score) + N x log of the base model's chance that a new word follows
           the candidate (ends_word: 在选 0.13, 再选 0.71); end0 is the rescoring alone
"""

import argparse
import json
import os
import random
import re
import sys

import jieba
import numpy as np
from pypinyin import Style, lazy_pinyin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import corpus  # noqa: E402
from ime import engine as engines  # noqa: E402
from ime.llm import LLMEngine  # noqa: E402

HAN = re.compile(r"^[一-鿿]+$")
HAN_CH = re.compile(r"[一-鿿]")
CASES = [("还有只修", "成语"), ("我们学了很多", "成语")]  # the user's case (10-19) and a context where it is easy
TOP_WORDS = 20000


def dict_words():
    rows = [l.split() for l in open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8")]
    rows = sorted(((w, int(f)) for w, f, *_ in rows if HAN.match(w)), key=lambda x: -x[1])
    return {w for w, _ in rows[:TOP_WORDS] if len(w) >= 2}, {w for w, _ in rows if len(w) == 2}


def keys(text, initials=False):
    p = [s.replace("ü", "v") for s in lazy_pinyin(text, style=Style.NORMAL, v_to_u=False)]
    return "".join(s[0] for s in p) if initials else "".join(p)


def items(n, root, common, pairs=None, seed=7):
    """n (context, word); with `pairs`, only words whose first character makes one of `pairs` with the context's last
    (还有只修 + 成语: 修成)."""
    rng = random.Random(seed)

    def pick(before, line):
        words = jieba.lcut(line)
        at, ok = 0, []
        for i, w in enumerate(words):
            if i and 2 <= len(w) <= 4 and w in common and len(HAN_CH.findall(line[:at])) >= 2 \
                    and (pairs is None or line[at - 1] + w[0] in pairs):
                ok.append((at, w))
            at += len(w)
        if not ok:
            return None
        at, w = rng.choice(ok)
        return ((before + line[:at])[-64:], w)

    chat, wiki = [], []
    for u in corpus.lccc(os.path.join(root, corpus.LCCC_TEST)):
        if rng.random() < (0.2 if pairs is None else 1) and (x := pick("", u)):
            chat.append(x)
        if len(chat) >= n * 4:
            break
    corpus.WIKI = os.path.join(root, corpus.WIKI)
    for _, text in corpus.wiki_articles(held=True):
        text = text[:2000]
        lines = [m for m in re.finditer(r"[^\n。！？]+", text) if m.start() > 20]
        if lines and (x := pick(text[: (m := rng.choice(lines)).start()], m.group())):
            wiki.append(x)
        if len(wiki) >= n * 4:
            break
    return rng.sample(chat, n // 2) + rng.sample(wiki, n - n // 2)


def cuts(text):
    at, out = 0, set()
    for w in jieba.lcut(text):
        at += len(w)
        out.add(at)
    return out


def ends_word(eng, context, cands, top=64):
    """For each candidate, the base model's chance that what follows it starts a new word: its `top` next tokens
    after context + candidate, those that jieba keeps apart from the candidate's last characters (or that are not Han:
    punctuation, the end) over all of them."""
    import mlx.core as mx

    model = eng.mix.model if eng.mix else eng.model
    plain = context[-64:] if context.strip() else eng.empty_prompt
    out = []
    for c in cands:
        logits = model(mx.array(eng.tok.encode(plain + c))[None])[0, -1]
        p = np.array(mx.softmax(logits.astype(mx.float32)))
        best = np.argsort(-p)[:top]
        tail = (plain + c)[-6:]
        apart = 0.0
        for t in best:
            s = eng.tok.decode([int(t)])
            if not s or not HAN_CH.match(s[0]) or len(tail) in cuts(tail + s):
                apart += p[t]
        out.append(max(apart / p[best].sum(), 1e-3))
    return np.array(out)


def patch(eng, variant, words, pairs):
    """eng with `variant` patched into its edges (the search's token costs)."""
    kind, nats = re.match(r"([a-z]+)([\d.]*)$", variant).groups()
    nats = float(nats or 0)
    if kind == "base":
        return
    if kind == "end":
        convert0 = LLMEngine.convert.__get__(eng)

        def conv_end(context, keys, k=5, **kw):
            cands = convert0(context, keys, k=max(10, k), **kw)
            if not cands or len(cands) < 2:
                return cands[:k] if cands else cands
            s = np.array(eng.score(context, keys, cands)) + nats * np.log(ends_word(eng, context, cands))
            return [cands[i] for i in np.argsort(-s, kind="stable")][:k]

        eng.convert = conv_end
        return
    edges = LLMEngine.edges.__get__(eng)
    multi = np.zeros(max(eng.tok_text) + 1, dtype=bool)
    if kind == "word":
        for t, s in eng.tok_text.items():
            multi[t] = len(s) >= 2 and s in words

    def patched(keys, p, swaps=True, alts=True):
        ends, tids, costs, fixes = edges(keys, p, swaps, alts)
        costs = costs.copy()
        if kind == "tok":
            costs += nats
        elif kind == "word":
            costs -= nats * multi[tids]
        elif kind == "first" and p == 0 and eng._last_ctx:
            for i, t in enumerate(tids):
                s = eng.tok_text[int(t)]
                if len(s) == 1 and eng._last_ctx + s in pairs:
                    costs[i] += nats
        return ends, tids, costs, fixes

    eng.edges = patched
    convert = LLMEngine.convert.__get__(eng)

    def conv(context, keys, **kw):
        eng._last_ctx = context[-1:] if context and HAN_CH.match(context[-1]) else ""
        return convert(context, keys, **kw)

    eng.convert = conv


def score(eng, rows):
    p1 = p5 = 0
    out = []
    for ctx, w, k in rows:
        c = eng.convert(ctx, k, k=5)
        p1 += bool(c) and c[0] == w
        p5 += w in c
        out.append(c)
    return p1, p5, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/r6-4bit")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--variants", default="base,tok1,tok2,word2,word4,first3")
    ap.add_argument("--initials", action="store_true")
    ap.add_argument("--sentences", action="store_true")
    ap.add_argument("--root", default=os.path.expanduser("~/Projects/llm-ime"), help="where data/corpus is")
    ap.add_argument("--show", type=int, default=8, help="items to print that the first variant misses")
    a = ap.parse_args()
    common, pairs = dict_words()
    sets = {}
    for name, its in (("word", items(a.n, a.root, common)), ("cross", items(a.n, a.root, common, pairs))):
        sets[f"{name} full"] = [(c, w, keys(w)) for c, w in its]
        if a.initials:
            sets[f"{name} initials"] = [(c, w, keys(w, True)) for c, w in its]
    if a.sentences:
        pub = [json.loads(l) for l in open(os.path.join(a.root, "data/private/eval.jsonl"), encoding="utf-8")]
        pub = [r for r in pub if r["src"] == "public"]
        for form in ("full", "initials"):
            sets[f"sentence {form}"] = [(r["context"], r["target"], r[form]) for r in pub]
    cases = [(c, w, keys(w)) for c, w in CASES]
    base_out = {}
    for v in a.variants.split(","):
        eng = engines.load("v02", model=a.model)
        patch(eng, v, common, pairs)
        _, _, cs = score(eng, cases)
        print(f"{v:8} cases: " + " | ".join(f"{c}+{k}: {' '.join(x)}" for (c, _w, k), x in zip(cases, cs)), flush=True)
        for name, rows in sets.items():
            p1, p5, out = score(eng, rows)
            print(f"{v:8} {name:18} n={len(rows)} P@1 {p1 / len(rows):.3f} P@5 {p5 / len(rows):.3f}", flush=True)
            if name not in base_out:
                base_out[name] = out
                miss = [(r, o) for r, o in zip(rows, out) if not o or o[0] != r[1]]
                for (c, w, k), o in miss[: a.show]:
                    print(f"    miss [{c[-16:]}] {w} ({k}): {' '.join(o)}")
            else:
                won = sum(o[:1] == [r[1]] and b[:1] != [r[1]] for r, o, b in zip(rows, out, base_out[name]))
                lost = sum(o[:1] != [r[1]] and b[:1] == [r[1]] for r, o, b in zip(rows, out, base_out[name]))
                print(f"         first vs {a.variants.split(',')[0]}: +{won} -{lost}", flush=True)
        del eng


if __name__ == "__main__":
    main()
