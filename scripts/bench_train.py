"""Time a few training steps under different settings (runs on dgx)."""

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "scripts")
import train  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.5)
torch.backends.cuda.matmul.allow_tf32 = True
data = train.load("data/train/train.npz")
configs = [(True, torch.float32, 32768), (False, torch.float32, 16384), (False, torch.bfloat16, 16384),
           (True, torch.bfloat16, 65536)]
for ckpt, dtype, tokens in configs:
    model = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B-Base", dtype=dtype, attn_implementation="sdpa").cuda()
    model.train()
    if ckpt:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)
    torch.cuda.reset_peak_memory_stats()
    it = train.batches(data, tokens, np.random.default_rng(0), model.config.eos_token_id)
    times, ntok = [], 0
    try:
        for i in range(8):
            x, y, att = (t.cuda() for t in next(it))
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = train.loss_fn(model, x, y, att)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.time() - t0)
                ntok += int(att.sum())
    except torch.OutOfMemoryError:
        print(f"ckpt={ckpt} {str(dtype)[6:]} tokens={tokens}: OOM", flush=True)
        del model, opt
        torch.cuda.empty_cache()
        continue
    print(f"ckpt={ckpt} {str(dtype)[6:]} tokens={tokens}: {np.mean(times):.2f} s/step, {ntok / sum(times):.0f} tok/s, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
