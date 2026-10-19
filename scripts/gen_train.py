"""Training samples for the conversion model → data/train/{train,valid}.npz (runs on dgx).

From each corpus line: pick a cut point, the target is the next 2–16 Han characters, the context is the
text before it, i.e. what an IME sees left of the cursor: the same message for chat, the article so far for
Wikipedia (dropped to empty 15% of the time). Keys: full / initials / mixed with equal weight; a character with
EXTRA_READINGS (嗯) is typed by any of its readings (n or en) alike.
6B.1: a line with English words between its Han characters (Wikipedia's 使用 Python 编写) gives, --english of the
time, a target with one or more of them instead (使用python编写: English lowercase, no spaces next to it, as the input
method writes the letters typed), the word typed as it is among the pinyin; --typos of the samples (I-1) have one
syllable of two keys or more typed with one key too many, missing, or next to the right one (not its first key, as
the typing log shows them, 5B.8), the target unchanged.
6B.6: --align of the targets start and end at a word boundary of jieba (the user's commits do, 94%), else anywhere in the
run of Han characters: a model trained with random cuts learns that the context may stop in the middle of a word (修|成)
and that the target may too (在选|择), and ranks 成与 above 成语 after 修.
Sources: LCCC and Wikipedia ~1:1, plus the user's own WeChat messages (--private) at ~5% of samples.
Stored pre-tokenized: ids (flat int32), offsets, prefix lengths (where the target starts) and the context's
token span cstart:cend (the context tokens get a down-weighted LM loss in train.py). --order picks the
prompt order (ime.fmt); the model's `ime_format` file must say the same.
"""

import argparse
import json
import os
import random
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import corpus, fmt, pinyin  # noqa: E402
from ime.ngram import with_extra  # noqa: E402

HAN_RUN = re.compile(r"[一-鿿]{2,}")
MIXED_RUN = re.compile(r"(?:[一-鿿]+|[A-Za-z]+| )+")
PART = re.compile(r"[一-鿿]|[A-Za-z]+")
TOK = None
ENGLISH, TYPOS, ALIGN = 0.8, 0.1, 0.0  # set by main


def mixed_target(line, rng):
    """(start in line, target, parts) of a span of Han characters with English words between them, or None. parts:
    one per character, one per English word (lowercase); two English words never next to each other (the space
    between them would be lost)."""
    spans = []
    for m in MIXED_RUN.finditer(line):
        parts = [(x.start() + m.start(), x.group()) for x in PART.finditer(m.group())]
        if any(p.isascii() for _, p in parts) and sum(not p.isascii() for _, p in parts) >= 2:
            spans.append(parts)
    if not spans:
        return None
    parts = rng.choice(spans)
    en = [i for i, (_, p) in enumerate(parts) if p.isascii() and 2 <= len(p) <= 20]
    if not en:
        return None
    i = rng.choice(en)
    lo = rng.randint(max(0, i - 8), i)
    hi = rng.randint(i + 1, min(len(parts), i + 9))
    win = parts[lo:hi]
    if sum(not p.isascii() for _, p in win) < 1 or len(win) < 2:
        return None
    if any(a.isascii() and b.isascii() for (_, a), (_, b) in zip(win, win[1:])):
        return None
    if any(p.isascii() and not 2 <= len(p) <= 20 for _, p in win):
        return None
    win = [(at, p.lower() if p.isascii() else p) for at, p in win]
    return win[0][0], "".join(p for _, p in win), [p for _, p in win]


ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")  # as ime.llm, which needs MLX (not on dgx)
KEY_AT = {c: (y, x + (0, 0.25, 0.75)[y]) for y, row in enumerate(ROWS) for x, c in enumerate(row)}


def adjacent(a, b):
    (ya, xa), (yb, xb) = KEY_AT[a], KEY_AT[b]
    return a != b and (abs(xa - xb) == 1 if ya == yb else abs(ya - yb) == 1 and abs(xa - xb) <= 0.75)


def typo(pieces, rng):
    """pieces with one key off in one of them of two keys or more (not its first key), or None if none can be."""
    where = [(i, j) for i, s in enumerate(pieces) for j in range(1, len(s)) if len(s) >= 2 and s.isalpha()]
    if not where:
        return None
    i, j = rng.choice(where)
    s, kind = pieces[i], rng.choice(("extra", "miss", "near"))
    if kind == "extra":
        s = s[:j] + rng.choice([s[j - 1]] + [k for k in "".join(ROWS) if adjacent(k, s[j - 1]) or adjacent(k, s[j])]) + s[j:]
    elif kind == "miss":
        s = s[:j] + s[j + 1 :]
    else:
        s = s[:j] + rng.choice([k for k in "".join(ROWS) if adjacent(k, s[j])]) + s[j + 1 :]
    return pieces[:i] + [s] + pieces[i + 1 :]


def init(tok_path, english=None, typos=None, align=None):
    global TOK, ENGLISH, TYPOS, ALIGN
    from transformers import AutoTokenizer

    if english is not None:
        ENGLISH, TYPOS, ALIGN = english, typos, align
    TOK = AutoTokenizer.from_pretrained(tok_path)


def word_cut(run, rng):
    """(start, end) in the Han run `run`, 2 to 16 characters, both at a word boundary of jieba, or None when no such
    cut exists (a word longer than 16 characters)."""
    import jieba

    bounds, at = [0], 0
    for w in jieba.lcut(run):
        at += len(w)
        bounds.append(at)
    starts = bounds[:-1]
    rng.shuffle(starts)
    for start in starts:
        ends = [b for b in bounds if start + 2 <= b <= start + 16]
        if ends:
            want = start + rng.randint(2, 16)  # the length as the random cut draws it, then the boundary at or before it
            return start, max([b for b in ends if b <= want] or [min(ends)])
    return None


def samples(lines, seed):
    """lines: (text before the line, line)."""
    rng = random.Random(seed)
    out = []
    for before, line in lines:
        mixed = mixed_target(line, rng) if rng.random() < ENGLISH else None
        if mixed:
            start, target, parts = mixed
            han = "".join(p for p in parts if not p.isascii())
            sy = iter(rng.choice(with_extra(han, pinyin.syllables(han))))
            s = [p if p.isascii() else next(sy) for p in parts]
            en = [p.isascii() for p in parts]
        else:
            runs = list(HAN_RUN.finditer(line))
            if not runs:
                continue
            m = rng.choice(runs)
            cut = word_cut(m.group(), rng) if rng.random() < ALIGN else None
            if cut:
                start, L = m.start() + cut[0], cut[1] - cut[0]
            else:
                start = rng.randrange(m.start(), m.end() - 1)
                L = min(rng.randint(2, 16), m.end() - start)
            if L < 2:
                continue
            target = line[start : start + L]
            s = rng.choice(with_extra(target, pinyin.syllables(target)))  # 嗯 typed n or en (ngram.EXTRA_READINGS)
            en = [False] * len(s)
        ctx = "" if rng.random() < 0.15 else (before + line[:start])[-fmt.CTX_CHARS :]
        form = rng.random()
        if form < 1 / 3:
            pieces = list(s)
        elif form < 2 / 3:
            pieces = [x if e else x[0] for x, e in zip(s, en)]
        else:  # English words stay whole
            han = iter(pinyin.mixed_parts([x for x, e in zip(s, en) if not e], rng))
            pieces = [x if e else next(han) for x, e in zip(s, en)]
        if rng.random() < TYPOS:
            off = typo([x for x, e in zip(pieces, en) if not e], rng)
            if off:
                off = iter(off)
                pieces = [x if e else next(off) for x, e in zip(pieces, en)]
        out.append((ctx, "".join(pieces), target, pieces))
    return out


def encode(args):
    lines, seed, order = args
    rows = []
    for ctx, keys, target, pieces in samples(lines, seed):
        if order == "interleave":
            if any(len(TOK.encode(ch)) != 1 for ch in target):  # rare characters: not one token each
                continue
            ids, p = fmt.interleave(TOK.encode, ctx, pieces, target)
            rows.append((ids + [TOK.eos_token_id], p, 0, p - len(TOK.encode(fmt.SEP1))))
            continue
        p = TOK.encode(fmt.prompt(ctx, keys, order))
        t = TOK.encode(target) + [TOK.eos_token_id]
        cs = ce = 0
        if ctx:
            if order == "keys_first":
                k = TOK.encode(fmt.keys_part(keys))
                if p[: len(k)] == k:
                    cs, ce = len(k), len(p)
            else:
                c = TOK.encode(ctx[-fmt.CTX_CHARS :])
                if p[: len(c)] == c:
                    cs, ce = 0, len(c)
        rows.append((p + t, len(p), cs, ce))
    return rows


def chunks(it, order, n=5000):
    buf, i = [], 0
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf, i, order
            buf, i = [], i + 1
    if buf:
        yield buf, i, order


EN_IN_HAN = re.compile(r"[一-鿿] ?[A-Za-z]{2,20} ?[一-鿿]")


def source(rng, private=None, english_extra=0.0):
    """(text before, line): LCCC utterances and Wikipedia sentences 1:1; private lines at ~5% of the output; with
    english_extra, as many more Wikipedia sentences with an English word between Han characters per LCCC line."""
    w = corpus.wiki_with_prefix(max_chars=400_000_000)
    en = (x for x in corpus.wiki_with_prefix(max_chars=10**12) if EN_IN_HAN.search(x[1])) if english_extra else None
    for u in corpus.lccc():
        yield "", u
        s = next(w, None)
        if s is not None:
            yield s
        if private and rng.random() < 0.105:
            yield "", rng.choice(private)
        if en is not None and rng.random() < english_extra:
            s = next(en, None)
            if s is not None:
                yield s


def save(path, rows, keyids=None):
    ids = np.concatenate([np.array(r[0], dtype=np.int32) for r in rows])
    lens = np.array([len(r[0]) for r in rows], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    plen = np.array([r[1] for r in rows], dtype=np.int32)
    cstart = np.array([r[2] for r in rows], dtype=np.int32)
    cend = np.array([r[3] for r in rows], dtype=np.int32)
    extra = {} if keyids is None else {"keyids": np.array(keyids, dtype=np.int32)}
    np.savez(path, ids=ids, offsets=offsets, plen=plen, cstart=cstart, cend=cend, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--tokenizer", default="models/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--order", default="ctx_first", choices=fmt.ORDERS)
    ap.add_argument("--private", default="", help="jsonl with a 'text' field (the user's own messages)")
    ap.add_argument("--english", type=float, default=0.8, help="share of lines with English words giving a mixed target")
    ap.add_argument("--english-extra", type=float, default=0.25,
                    help="more Wikipedia lines with English between Han characters, per LCCC line")
    ap.add_argument("--typos", type=float, default=0.1, help="share of samples with one key off (I-1)")
    ap.add_argument("--align", type=float, default=0.0, help="share of targets cut at jieba word boundaries (6B.6)")
    a = ap.parse_args()
    global ENGLISH, TYPOS, ALIGN
    ENGLISH, TYPOS, ALIGN = (0.0 if a.order == "interleave" else a.english), a.typos, a.align
    private = [json.loads(x)["text"] for x in open(a.private)] if a.private else None
    rows = []
    rng = random.Random(0)
    with Pool(os.cpu_count() - 2, initializer=init, initargs=(a.tokenizer, ENGLISH, TYPOS, ALIGN)) as pool:
        for r in pool.imap(encode, chunks(source(rng, private, a.english_extra if ENGLISH else 0.0), a.order)):
            rows += r
            if len(rows) // 200_000 != (len(rows) - len(r)) // 200_000:
                print(f"  {len(rows)}", flush=True)
            if len(rows) >= a.n:
                break
    rng.shuffle(rows)
    os.makedirs(a.out, exist_ok=True)
    init(a.tokenizer)
    # interleave: the key tokens inside the target span (the letters) get no loss in train.py
    keyids = [TOK.encode(c)[0] for c in "abcdefghijklmnopqrstuvwxyz"] if a.order == "interleave" else None
    save(f"{a.out}/valid.npz", rows[:5000], keyids)
    save(f"{a.out}/train.npz", rows[5000 : a.n], keyids)
    toks = sum(len(r[0]) for r in rows)
    print(f"{len(rows)} samples, {toks / len(rows):.1f} tokens each")
    for r, p, cs, ce in rows[:5]:
        print(repr(TOK.decode(r[:p])), "[ctx:", repr(TOK.decode(r[cs:ce])), "] →", repr(TOK.decode(r[p:])))


if __name__ == "__main__":
    main()
