"""Training samples for the conversion model → data/train/{train,valid}.npz (runs on dgx).

From each corpus line: pick a cut point, the target is the next 2–16 Han characters, the context is the
text before it, i.e. what an IME sees left of the cursor: the same message for chat, the article so far for
Wikipedia (dropped to empty 15% of the time). Keys: full / initials / mixed with equal weight.
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

HAN_RUN = re.compile(r"[一-鿿]{2,}")
TOK = None


def init(tok_path):
    global TOK
    from transformers import AutoTokenizer

    TOK = AutoTokenizer.from_pretrained(tok_path)


def samples(lines, seed):
    """lines: (text before the line, line)."""
    rng = random.Random(seed)
    out = []
    for before, line in lines:
        runs = list(HAN_RUN.finditer(line))
        if not runs:
            continue
        m = rng.choice(runs)
        start = rng.randrange(m.start(), m.end() - 1)
        L = min(rng.randint(2, 16), m.end() - start)
        if L < 2:
            continue
        target = line[start : start + L]
        ctx = "" if rng.random() < 0.15 else (before + line[:start])[-fmt.CTX_CHARS :]
        s = pinyin.syllables(target)
        form = rng.random()
        keys = pinyin.full(s) if form < 1 / 3 else pinyin.initials(s) if form < 2 / 3 else pinyin.mixed(s, rng)
        out.append((ctx, keys, target))
    return out


def encode(args):
    lines, seed, order = args
    rows = []
    for ctx, keys, target in samples(lines, seed):
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


def source(rng, private=None):
    """(text before, line): LCCC utterances and Wikipedia sentences 1:1; private lines at ~5% of the output."""
    w = corpus.wiki_with_prefix(max_chars=400_000_000)
    for u in corpus.lccc():
        yield "", u
        s = next(w, None)
        if s is not None:
            yield s
        if private and rng.random() < 0.105:
            yield "", rng.choice(private)


def save(path, rows):
    ids = np.concatenate([np.array(r[0], dtype=np.int32) for r in rows])
    lens = np.array([len(r[0]) for r in rows], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    plen = np.array([r[1] for r in rows], dtype=np.int32)
    cstart = np.array([r[2] for r in rows], dtype=np.int32)
    cend = np.array([r[3] for r in rows], dtype=np.int32)
    np.savez(path, ids=ids, offsets=offsets, plen=plen, cstart=cstart, cend=cend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--tokenizer", default="models/Qwen3-0.6B-Base")
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--order", default="ctx_first", choices=fmt.ORDERS)
    ap.add_argument("--private", default="", help="jsonl with a 'text' field (the user's own messages)")
    a = ap.parse_args()
    private = [json.loads(x)["text"] for x in open(a.private)] if a.private else None
    rows = []
    rng = random.Random(0)
    with Pool(os.cpu_count() - 2, initializer=init, initargs=(a.tokenizer,)) as pool:
        for r in pool.imap(encode, chunks(source(rng, private), a.order)):
            rows += r
            if len(rows) // 200_000 != (len(rows) - len(r)) // 200_000:
                print(f"  {len(rows)}", flush=True)
            if len(rows) >= a.n:
                break
    rng.shuffle(rows)
    os.makedirs(a.out, exist_ok=True)
    save(f"{a.out}/valid.npz", rows[:5000])
    save(f"{a.out}/train.npz", rows[5000 : a.n])
    toks = sum(len(r[0]) for r in rows)
    print(f"{len(rows)} samples, {toks / len(rows):.1f} tokens each")
    init(a.tokenizer)
    for r, p, cs, ce in rows[:5]:
        print(repr(TOK.decode(r[:p])), "[ctx:", repr(TOK.decode(r[cs:ce])), "] →", repr(TOK.decode(r[p:])))


if __name__ == "__main__":
    main()
