"""Per token log-probabilities of texts for keys after a context, model by model (6B.7: 及格线 is 30 nats behind 几个线
for r6 after 「只是到了」).

  .venv/bin/python scripts/token_scores.py --model models/r6-4bit --model models/r7-4bit --keys jigexian \
      --text 及格线 --text 几个线 --context 全身插画那个只是到了

Per model: the fine-tuned model's own log-probabilities (the keys in its prompt) and, from the base model mixed in (v02's
mix, weight 1.0 with a context), its log-probabilities after the context alone; the rank and the best token of the first
step, and a sum as the search adds it (fine-tuned with its end-of-sentence score as the last number + mix). A model named
models/qwen3-0.6b-base-4bit is scored as a base model only.
"""

import argparse
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime import fmt  # noqa: E402


def run(eng, ctx, keys, texts, base_only):
    rows = []
    plain = ctx[-64:] if ctx.strip() else eng.empty_prompt
    if not base_only:
        eng._prefill(fmt.prompt(ctx, keys, eng.order))
    eng.mix.prefill(eng.tok.encode(plain))
    for text in texts:
        ids = eng.tok.encode(text)
        out = {}
        for name, model, first, mix in (("ft", eng.model, None if base_only else eng._ctx_logits, False),
                                        ("base", eng.mix.model, eng.mix.ctx_logits, True)):
            if first is None:
                continue
            cache = eng.mix.fork(mx.array([0]), len(ids) + 2) if mix else eng._fork_cache(mx.array([0]), len(ids) + 2)
            l, lp, rank, best = first, [], None, None
            for i, t in enumerate(ids):
                lps = l - mx.logsumexp(l, axis=-1, keepdims=True)
                lp.append(round(float(lps[0, t]), 1))
                if i == 0:
                    rank = int(mx.sum(l[0] > l[0, t]).item()) + 1
                    best = eng.tok.decode([int(mx.argmax(l[0]).item())])
                l = model(mx.array([[t]]), cache=cache)[:, -1, :]
            if name == "ft":  # the fine-tuned model scores the end of the sentence too
                lp.append(round(float((l - mx.logsumexp(l, axis=-1, keepdims=True))[0, eng.eos]), 1))
            out[name] = (lp, rank, best)
        rows.append((text, [eng.tok.decode([t]) for t in ids], out))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--text", action="append", required=True)
    ap.add_argument("--context", action="append", default=[])
    a = ap.parse_args()
    for m in a.model:
        base_only = "qwen3-0.6b-base" in m
        eng = engines.load("v02", model=m, mix="models/qwen3-0.6b-base-4bit") if not base_only else \
            engines.load("v02", model="models/r6-4bit")
        if base_only:
            eng.model = eng.mix.model  # only the base model is read
        for ctx in a.context or [""]:
            for text, toks, out in run(eng, ctx, a.keys, a.text, base_only):
                parts = []
                for name, (lp, rank, best) in out.items():
                    parts.append(f"{name} {lp} sum {sum(lp):.1f} first-token rank {rank} (best {best})")
                total = sum(sum(v[0]) for v in out.values()) if not base_only else sum(out["base"][0])
                print(f"{os.path.basename(m):22} [{ctx[-8:]}] {text} {toks}: " + " | ".join(parts) + f" | ft+mix {total:.1f}")


if __name__ == "__main__":
    main()
