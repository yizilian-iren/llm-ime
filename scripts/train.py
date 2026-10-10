"""Full fine-tune of a causal LM on data/train/*.npz (runs on dgx).

Batches are built by token budget from length-sorted buckets, right-padded; loss only on target tokens.

  .venv-train/bin/python scripts/train.py --out runs/r1 [--max-steps 200]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM


def load(path):
    d = np.load(path)
    return d["ids"], d["offsets"], d["plen"]


def batches(data, tokens_per_batch, rng, pad_id):
    ids, off, plen = data
    lens = np.diff(off)
    order = rng.permutation(len(lens))
    # bucket into chunks of 100 batches' worth, sort within chunk by length to limit padding
    chunk = 256 * 100
    for c in range(0, len(order), chunk):
        idx = order[c : c + chunk]
        idx = idx[np.argsort(lens[idx], kind="stable")]
        groups, cur, cur_max = [], [], 0
        for i in idx:
            m = max(cur_max, lens[i])
            if cur and m * (len(cur) + 1) > tokens_per_batch:
                groups.append(cur)
                cur, m = [], lens[i]
            cur.append(i)
            cur_max = m
        if cur:
            groups.append(cur)
        rng.shuffle(groups)
        for g in groups:
            T = int(max(lens[i] for i in g))
            x = np.full((len(g), T), pad_id, dtype=np.int64)
            y = np.full((len(g), T), -100, dtype=np.int64)
            att = np.zeros((len(g), T), dtype=np.int64)
            for r, i in enumerate(g):
                seq = ids[off[i] : off[i + 1]]
                x[r, : len(seq)] = seq
                att[r, : len(seq)] = 1
                y[r, plen[i] : len(seq)] = seq[plen[i] :]
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(att)


def loss_fn(model, x, y, att):
    # Only target positions go through lm_head: full-vocab logits for every token would be tens of GB.
    # Chunked so the (targets x 151k vocab) logits never exist all at once.
    h = model.model(input_ids=x, attention_mask=att).last_hidden_state[:, :-1]
    tgt = y[:, 1:]
    mask = tgt != -100
    h, tgt = h[mask], tgt[mask]
    total = 0.0
    for i in range(0, len(tgt), 2048):
        total = total + torch.utils.checkpoint.checkpoint(_ce_chunk, model.lm_head, h[i : i + 2048], tgt[i : i + 2048],
                                                          use_reentrant=False)
    return total / len(tgt)


def _ce_chunk(head, h, t):
    return F.cross_entropy(head(h).float(), t, reduction="sum")


@torch.no_grad()
def evaluate(model, data, pad_id, dev):
    model.eval()
    tot, n = 0.0, 0
    for x, y, att in batches(data, 16384, np.random.default_rng(0), pad_id):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model, x.to(dev), y.to(dev), att.to(dev))
        k = int((y[:, 1:] != -100).sum())
        tot += float(loss) * k
        n += k
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tokens", type=int, default=16384, help="tokens per batch (padded)")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing: ~18 GB instead of ~52 GB, but 20%% slower")
    ap.add_argument("--mem-frac", type=float, default=0.5, help="cap on this process's share of the shared GPU memory")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    # dgx is shared and its memory is unified with the CPU: fail with OOM rather than freeze the machine.
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="sdpa").to(dev)
    model.train()
    if a.ckpt:
        model.gradient_checkpointing_enable()
    pad_id = model.config.eos_token_id
    train, valid = load("data/train/train.npz"), load("data/train/valid.npz")
    n_tok = int(train[1][-1])
    steps = a.max_steps or int(a.epochs * n_tok / (a.tokens * 0.8))
    print(f"{len(train[2])} samples, {n_tok / 1e6:.0f}M tokens, ~{steps} steps", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rng = np.random.default_rng(1)
    step, t0, seen = 0, time.time(), 0
    print(f"step 0 valid {evaluate(model, valid, pad_id, dev):.4f}", flush=True)
    while step < steps:
        for x, y, att in batches(train, a.tokens, rng, pad_id):
            x, y, att = x.to(dev), y.to(dev), att.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model, x, y, att)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += int(att.sum())
            if step % 50 == 0:
                rate = seen / (time.time() - t0)
                rec = {"step": step, "loss": float(loss), "lr": sched.get_last_lr()[0], "tok_s": round(rate),
                       "eta_h": round((steps - step) * a.tokens * 0.8 / rate / 3600, 2)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            if step % a.eval_every == 0 or step == steps:
                v = evaluate(model, valid, pad_id, dev)
                log.write(json.dumps({"step": step, "valid": v}) + "\n")
                log.flush()
                print(f"step {step} valid {v:.4f}", flush=True)
                model.save_pretrained(os.path.join(a.out, "ckpt"), safe_serialization=True)
            if step >= steps:
                break
    model.to(torch.bfloat16).save_pretrained(os.path.join(a.out, "final"), safe_serialization=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
