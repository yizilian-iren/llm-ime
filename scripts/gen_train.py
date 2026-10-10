"""Training samples for the conversion model → data/train/{train,valid}.npz (runs on dgx).

From each corpus line: pick a cut point, the target is the next 2–16 Han characters, the context is the
text before it (dropped to empty 30% of the time). Keys: full / initials / mixed with equal weight.
Stored pre-tokenized: ids (flat int32), offsets, and prefix lengths (loss only after SEP2).
"""

import argparse
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
    rng = random.Random(seed)
    out = []
    for line in lines:
        runs = list(HAN_RUN.finditer(line))
        if not runs:
            continue
        m = rng.choice(runs)
        start = rng.randrange(m.start(), m.end() - 1)
        L = min(rng.randint(2, 16), m.end() - start)
        if L < 2:
            continue
        target = line[start : start + L]
        ctx = "" if rng.random() < 0.3 else line[max(0, start - fmt.CTX_CHARS) : start]
        s = pinyin.syllables(target)
        form = rng.random()
        keys = pinyin.full(s) if form < 1 / 3 else pinyin.initials(s) if form < 2 / 3 else pinyin.mixed(s, rng)
        out.append((ctx, keys, target))
    return out


def encode(args):
    lines, seed = args
    rows = []
    for ctx, keys, target in samples(lines, seed):
        p = TOK.encode(fmt.prompt(ctx, keys))
        t = TOK.encode(target) + [TOK.eos_token_id]
        rows.append((p + t, len(p)))
    return rows


def chunks(it, n=5000):
    buf, i = [], 0
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf, i
            buf, i = [], i + 1
    if buf:
        yield buf, i


def source(rng):
    """LCCC utterances and Wikipedia sentences interleaved ~1:1."""
    w = corpus.wiki(max_chars=400_000_000)
    for u in corpus.lccc():
        yield u
        if rng.random() < 0.5:
            s = next(w, None)
            if s is not None:
                yield s


def save(path, rows):
    ids = np.concatenate([np.array(r, dtype=np.int32) for r, _ in rows])
    lens = np.array([len(r) for r, _ in rows], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    plen = np.array([p for _, p in rows], dtype=np.int32)
    np.savez(path, ids=ids, offsets=offsets, plen=plen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4_000_000)
    ap.add_argument("--tokenizer", default="models/Qwen3-0.6B-Base")
    a = ap.parse_args()
    rows = []
    rng = random.Random(0)
    with Pool(os.cpu_count() - 2, initializer=init, initargs=(a.tokenizer,)) as pool:
        for r in pool.imap(encode, chunks(source(rng))):
            rows += r
            if len(rows) // 200_000 != (len(rows) - len(r)) // 200_000:
                print(f"  {len(rows)}", flush=True)
            if len(rows) >= a.n:
                break
    rng.shuffle(rows)
    os.makedirs("data/train", exist_ok=True)
    save("data/train/valid.npz", rows[:5000])
    save("data/train/train.npz", rows[5000 : a.n])
    toks = sum(len(r) for r, _ in rows)
    print(f"{len(rows)} samples, {toks / len(rows):.1f} tokens each")
    init(a.tokenizer)
    for r, p in rows[:5]:
        print(repr(TOK.decode(r[:p])), "→", repr(TOK.decode(r[p:])))


if __name__ == "__main__":
    main()
