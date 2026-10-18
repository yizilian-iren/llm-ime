"""The weekly retrain's samples (scripts/retrain_samples.py) as train.py's data (6A.2): runs where it trains.

  .venv-train/bin/python scripts/retrain_npz.py SAMPLES.jsonl --test-days 2026-10-17,2026-10-18 \
      --tokenizer models/r2-hf --order ctx_first --out DIR

DIR/train.npz from the other days, DIR/valid.npz from the test days, encoded as scripts/gen_train.py does
(the order: the base model's `ime_format`). Prints counts only.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import fmt  # noqa: E402  (only the standard library: the training machine has no jieba / pypinyin)


def save(path, rows, keyids=None):  # as scripts/gen_train.save, which imports the corpus tools
    ids = np.concatenate([np.array(r[0], dtype=np.int32) for r in rows])
    offsets = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])]).astype(np.int64)
    extra = {} if keyids is None else {"keyids": np.array(keyids, dtype=np.int32)}
    np.savez(path, ids=ids, offsets=offsets, plen=np.array([r[1] for r in rows], dtype=np.int32),
             cstart=np.array([r[2] for r in rows], dtype=np.int32), cend=np.array([r[3] for r in rows], dtype=np.int32),
             **extra)


def encode(tok, s, order):
    """(ids, prefix length, context start, context end) or None, as gen_train.encode."""
    ctx, keys, target = s["context"], s["keys"], s["text"]
    if order == "interleave":
        if any(len(tok.encode(ch)) != 1 for ch in target):
            return None
        ids, p = fmt.interleave(tok.encode, ctx, s["pieces"], target)
        return ids + [tok.eos_token_id], p, 0, p - len(tok.encode(fmt.SEP1))
    p = tok.encode(fmt.prompt(ctx, keys, order))
    t = tok.encode(target) + [tok.eos_token_id]
    cs = ce = 0
    if ctx:
        if order == "keys_first":
            k = tok.encode(fmt.keys_part(keys))
            if p[: len(k)] == k:
                cs, ce = len(k), len(p)
        else:
            c = tok.encode(ctx[-fmt.CTX_CHARS :])
            if p[: len(c)] == c:
                cs, ce = 0, len(c)
    return p + t, len(p), cs, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("--test-days", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--order", default="ctx_first", choices=fmt.ORDERS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    test_days = set(a.test_days.split(","))
    rows = {"train": [], "valid": []}
    for line in open(a.samples, encoding="utf-8"):
        s = json.loads(line)
        if "counts" in s:
            continue
        r = encode(tok, s, a.order)
        if r:
            rows["valid" if s["day"] in test_days else "train"].append(r)
    os.makedirs(a.out, exist_ok=True)
    keyids = [tok.encode(c)[0] for c in "abcdefghijklmnopqrstuvwxyz"] if a.order == "interleave" else None
    for name, rs in rows.items():
        if not rs:
            sys.exit(f"no {name} samples")
        save(os.path.join(a.out, f"{name}.npz"), rs, keyids)
    tokens = {k: sum(len(r[0]) for r in v) for k, v in rows.items()}
    print(json.dumps({"train": len(rows["train"]), "valid": len(rows["valid"]), "tokens": tokens}))


if __name__ == "__main__":
    main()
