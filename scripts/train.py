"""Full fine-tune of a causal LM on data/train/*.npz (runs on dgx).

Batches are built by token budget from length-sorted buckets, right-padded. Loss on target tokens, plus the
context tokens at --ctx-weight (plain LM loss, so the model keeps its language ability); valid is target-only.

Distillation (--teacher, r4): on the target tokens, also a KL term to what the engine's mixed scoring would
give: log p_teacher(sees the keys) + w * log p_mix(base model, sees only the context and the target, as in
ime.llm), with w = --mix-weight (--mix-weight-empty with no context), end of sentence from the teacher only,
renormalised. The mix model's input is rebuilt from the sample: its context tokens (or "\n") + the target.

  .venv-train/bin/python scripts/train.py --out runs/r1 [--max-steps 200]
  .venv-train/bin/python scripts/train.py --model runs/r2/final --teacher runs/r2/final \
      --mix models/Qwen3-0.6B-Base --distill 1.0 --out runs/r4 --ckpt
  r4a2: --model runs/r4/final --ce 0.2 --lr 3e-5 (the plain loss at full weight pulled r4 back towards r2)

LoRA (--lora R, 6A.1: home4070's 4070 has ~5-6 GB left beside voice-tts): the model frozen in bf16, adapters of rank
R on every linear layer of the blocks; saves the adapter (final) and the model with it merged in (final-merged, for
the MLX conversion). Each log line has the peak GPU memory PyTorch allocated (mem_gb).
  .venv-train/bin/python scripts/train.py --model models/r2-hf --lora 16 --ckpt --tokens 8192 --lr 2e-4 --out runs/l1
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

CE_CHUNK = 2048  # target positions per lm_head call: 2048 x 151k fp32 logits are 1.2 GB (--ce-chunk)


def load(path):
    d = np.load(path)
    if "cstart" in d:
        cs, ce = d["cstart"], d["cend"]
    else:  # r2 data: context first, clen tokens long
        ce = d["clen"] if "clen" in d else np.zeros_like(d["plen"])
        cs = np.zeros_like(ce)
    keyids = d["keyids"] if "keyids" in d else None  # interleave data: letters in the target span, no loss
    return d["ids"], d["offsets"], d["plen"], cs, ce, keyids


def take(data, idx):
    """Rows idx of loaded data, as loaded data."""
    ids, off, plen, cs, ce, keyids = data
    parts = [ids[off[i] : off[i + 1]] for i in idx]
    offsets = np.concatenate([[0], np.cumsum([len(q) for q in parts])]).astype(off.dtype)
    return np.concatenate(parts), offsets, plen[idx], cs[idx], ce[idx], keyids


def concat(a, b):
    return (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1][1:] + a[1][-1]]),
            *(np.concatenate([x, y]) for x, y in zip(a[2:5], b[2:5])), a[5])


def batches(data, tokens_per_batch, rng, pad_id, ctx_weight=0.0, mix_input=None, key_weight=0.0):
    """mix_input: (newline token id, w with context, w without) to also build the mix model's input.
    key_weight: interleave data's letters in the target span get this weight (0 = no loss on them)."""
    ids, off, plen, cstart, cend, keyids = data
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
            w = np.zeros((len(g), T), dtype=np.float32)
            for r, i in enumerate(g):
                seq = ids[off[i] : off[i + 1]]
                x[r, : len(seq)] = seq
                att[r, : len(seq)] = 1
                y[r, plen[i] : len(seq)] = seq[plen[i] :]
                w[r, plen[i] : len(seq)] = 1.0
                if keyids is not None:
                    k = plen[i] + np.flatnonzero(np.isin(seq[plen[i] :], keyids))
                    if key_weight:
                        w[r, k] = key_weight
                    else:
                        y[r, k], w[r, k] = -100, 0.0
                a, b = max(1, cstart[i]), cend[i]  # position 0 has nothing before it to predict from
                if ctx_weight and b > a:
                    y[r, a:b] = seq[a:b]
                    w[r, a:b] = ctx_weight
            batch = [torch.from_numpy(a) for a in (x, y, att, w)]
            if mix_input:
                batch.append(_mix_batch(ids, off, plen, cstart, cend, g, pad_id, *mix_input))
            yield batch


def _mix_batch(ids, off, plen, cstart, cend, g, pad_id, nl_id, w_ctx, w_empty):
    """The mix model's input for rows g: context tokens (or a newline) + target, and for both inputs the
    positions that predict a target token (same count per row, so the flattened orders line up)."""
    seqs, tpos, bpos, wm = [], [], [], []
    for i in g:
        seq = ids[off[i] : off[i + 1]]
        p, cs, ce = int(plen[i]), int(cstart[i]), int(cend[i])
        ctx = list(seq[cs:ce]) if ce > cs else [nl_id]
        seqs.append(ctx + list(seq[p:]))
        n_t = len(seq) - p
        tpos.append(range(p - 1, p - 1 + n_t))
        bpos.append(range(len(ctx) - 1, len(ctx) - 1 + n_t))
        wm.append(w_ctx if ce > cs else w_empty)
    T = max(len(q) for q in seqs)
    xb = np.full((len(g), T), pad_id, dtype=np.int64)
    attb = np.zeros((len(g), T), dtype=np.int64)
    bmask = np.zeros((len(g), T), dtype=bool)
    tmask = np.zeros((len(g), int(max(off[i + 1] - off[i] for i in g))), dtype=bool)
    wrow = np.zeros(len(g), dtype=np.float32)
    for r, q in enumerate(seqs):
        xb[r, : len(q)] = q
        attb[r, : len(q)] = 1
        bmask[r, list(bpos[r])] = True
        tmask[r, list(tpos[r])] = True
        wrow[r] = wm[r]
    wpos = np.repeat(wrow, tmask.sum(1))  # mix weight per target position, in flattened order
    return [torch.from_numpy(a) for a in (xb, attb, tmask, bmask, wpos)]


def loss_fn(model, x, y, att, w, mix_batch=None, teachers=None, distill=0.0, eos_id=None, ce=1.0):
    # Only target positions go through lm_head: full-vocab logits for every token would be tens of GB.
    # Chunked so the (targets x 151k vocab) logits never exist all at once.
    # use_cache=False: else every layer keeps its keys and values, 1.9 GB at 16k tokens (6A.1)
    hs = model.model(input_ids=x, attention_mask=att, use_cache=False).last_hidden_state
    h = hs[:, :-1]
    tgt = y[:, 1:]
    mask = tgt != -100
    h, tgt, w = h[mask], tgt[mask], w[:, 1:][mask]
    total = 0.0
    for i in range(0, len(tgt), CE_CHUNK):
        j = i + CE_CHUNK
        total = total + torch.utils.checkpoint.checkpoint(_ce_chunk, model.lm_head, h[i:j], tgt[i:j], w[i:j],
                                                          use_reentrant=False)
    loss = total / w.sum()
    if not distill:
        return loss, None
    loss = ce * loss
    xb, attb, tmask, bmask, wpos = mix_batch
    teacher, mix = teachers
    hs_t = hs[:, : tmask.shape[1]][tmask]  # student, at the positions predicting target tokens
    with torch.no_grad():
        ht = teacher.model(input_ids=x, attention_mask=att, use_cache=False).last_hidden_state[:, : tmask.shape[1]][tmask]
        hb = mix.model(input_ids=xb, attention_mask=attb, use_cache=False).last_hidden_state[bmask]
    kl = 0.0
    for i in range(0, len(hs_t), 512):
        with torch.no_grad():
            lt = teacher.lm_head(ht[i : i + 512]).float().log_softmax(-1)
            comb = lt + wpos[i : i + 512, None] * mix.lm_head(hb[i : i + 512]).float().log_softmax(-1)
            comb[:, eos_id] = lt[:, eos_id]
            logq = comb.log_softmax(-1)
            del lt, comb
        kl = kl + torch.utils.checkpoint.checkpoint(_kl_chunk, model.lm_head, hs_t[i : i + 512], logq,
                                                    use_reentrant=False)
    kl = kl / len(hs_t)
    return loss + distill * kl, kl


def _ce_chunk(head, h, t, w):
    return (F.cross_entropy(head(h).float(), t, reduction="none") * w).sum()


def _kl_chunk(head, h, logq):
    return (logq.exp() * (logq - head(h).float().log_softmax(-1))).sum()


@torch.no_grad()
def evaluate(model, data, pad_id, dev):
    model.eval()
    tot, n = 0.0, 0
    for x, y, att, w in batches(data, 16384, np.random.default_rng(0), pad_id):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = loss_fn(model, x.to(dev), y.to(dev), att.to(dev), w.to(dev))
        k = int((y[:, 1:] != -100).sum())
        tot += float(loss) * k
        n += k
    model.train()
    return tot / n


def write_progress(out, done, total, eta_s, note, status="running"):
    """For Agent Deck's sidebar progress bar (background_task progress_file)."""
    tmp = os.path.join(out, "progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"done": done, "total": total, "eta_s": round(eta_s), "status": status, "note": note}, f)
    os.replace(tmp, os.path.join(out, "progress.json"))


def main():
    global CE_CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B-Base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--ctx-weight", type=float, default=0.3, help="LM loss weight on context tokens (0 = target only)")
    ap.add_argument("--tokens", type=int, default=16384, help="tokens per batch (padded)")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing: ~18 GB instead of ~52 GB, but 20%% slower")
    ap.add_argument("--mem-frac", type=float, default=0.5, help="cap on this process's share of the shared GPU memory")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--teacher", default="", help="distil from this model (sees the keys) mixed with --mix")
    ap.add_argument("--mix", default="", help="the base model mixed into the teacher's scores")
    ap.add_argument("--mix-weight", type=float, default=1.0)
    ap.add_argument("--mix-weight-empty", type=float, default=0.5)
    ap.add_argument("--distill", type=float, default=1.0, help="weight of the KL term (with --teacher)")
    ap.add_argument("--key-weight", type=float, default=0.0,
                    help="interleave data: loss weight on the letters, so the model scores keys too (r4b2: 1.0)")
    ap.add_argument("--ce", type=float, default=1.0, help="weight of the plain loss when distilling (r4a2: 0.2)")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0: full fine-tune)")
    ap.add_argument("--lora-alpha", type=float, default=0, help="default 2 x rank")
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--replay", default="", help="also train on rows of this npz (public data, so the weekly LoRA on "
                    "the user's commits does not forget the rest, 6A.2)")
    ap.add_argument("--replay-ratio", type=float, default=1.0, help="replayed rows per row of --data")
    ap.add_argument("--ce-chunk", type=int, default=CE_CHUNK, help="smaller: less memory for the logits (6A.1: 256)")
    a = ap.parse_args()
    CE_CHUNK = a.ce_chunk
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    # dgx is shared and its memory is unified with the CPU: fail with OOM rather than freeze the machine.
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)

    dtype = torch.bfloat16 if a.lora else torch.float32  # frozen weights need no fp32 master copy
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="sdpa").to(dev)
    model.train()
    if a.ckpt:  # non-reentrant: with LoRA the inputs of a checkpointed block need no grad of their own
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft = None
    if a.lora:
        from peft import LoraConfig, get_peft_model

        peft = get_peft_model(model, LoraConfig(r=a.lora, lora_alpha=a.lora_alpha or 2 * a.lora, lora_dropout=0.0,
                                                target_modules=a.lora_targets.split(",")))
        for p in model.parameters():  # adapters in fp32 under autocast
            if p.requires_grad:
                p.data = p.data.float()
        model = peft.get_base_model()  # the same modules, adapters injected: loss_fn uses model.model / lm_head
        peft.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    pad_id = model.config.eos_token_id
    teachers, mix_input, distill = None, None, 0.0
    if a.teacher:
        from transformers import AutoTokenizer

        teachers = [AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
                    for m in (a.teacher, a.mix)]
        for t in teachers:
            t.requires_grad_(False)
        nl = AutoTokenizer.from_pretrained(a.mix).encode("\n")
        assert len(nl) == 1, nl
        mix_input, distill = (nl[0], a.mix_weight, a.mix_weight_empty), a.distill
    train, valid = load(f"{a.data}/train.npz"), load(f"{a.data}/valid.npz")
    if a.replay:
        public = load(a.replay)
        n = min(len(public[2]), int(a.replay_ratio * len(train[2])))
        idx = np.sort(np.random.default_rng(2).choice(len(public[2]), n, replace=False))
        print(f"{len(train[2])} samples of --data, {n} replayed from {a.replay}", flush=True)
        train = concat(train, take(public, idx))
    assert not (a.teacher and train[5] is not None), "distillation is not written for interleave data"
    n_tok = int(train[1][-1])
    steps = a.max_steps or int(a.epochs * n_tok / (a.tokens * 0.8))
    print(f"{len(train[2])} samples, {n_tok / 1e6:.0f}M tokens, ~{steps} steps", flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rng = np.random.default_rng(1)
    step, t0, seen = 0, time.time(), 0
    print(f"step 0 valid {evaluate(model, valid, pad_id, dev):.4f}", flush=True)
    while step < steps:
        for x, y, att, w, *mb in batches(train, a.tokens, rng, pad_id, a.ctx_weight, mix_input, a.key_weight):
            x, y, att, w = x.to(dev), y.to(dev), att.to(dev), w.to(dev)
            mb = [t.to(dev) for t in mb[0]] if mb else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, kl = loss_fn(model, x, y, att, w, mb, teachers, distill, pad_id, a.ce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += int(att.sum())
            if step % 50 == 0:
                rate = seen / (time.time() - t0)
                rec = {"step": step, "loss": float(loss), **({"kl": round(float(kl), 4)} if kl is not None else {}),
                       "lr": sched.get_last_lr()[0], "tok_s": round(rate),
                       "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       "eta_h": round((steps - step) * a.tokens * 0.8 / rate / 3600, 2)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
                write_progress(a.out, step, steps, rec["eta_h"] * 3600, f"loss {rec['loss']:.3f}")
            if step % a.eval_every == 0 or step == steps:
                v = evaluate(model, valid, pad_id, dev)
                log.write(json.dumps({"step": step, "valid": v}) + "\n")
                log.flush()
                print(f"step {step} valid {v:.4f}", flush=True)
                (peft or model).save_pretrained(os.path.join(a.out, "ckpt"), safe_serialization=True)
            if step >= steps:
                break
    if peft:
        peft.save_pretrained(os.path.join(a.out, "final"), safe_serialization=True)
        peft.merge_and_unload().to(torch.bfloat16).save_pretrained(os.path.join(a.out, "final-merged"),
                                                                   safe_serialization=True)
    else:
        model.to(torch.bfloat16).save_pretrained(os.path.join(a.out, "final"), safe_serialization=True)
    write_progress(a.out, step, steps, 0, "saved final", "done")
    print("done", flush=True)


if __name__ == "__main__":
    main()
