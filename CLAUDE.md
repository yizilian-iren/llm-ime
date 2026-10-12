# llm-ime: pitfalls

- Training formats where the model decides as it reads (interleave): give the inputs it reads a loss too (`train.py --key-weight`), or the score never pays for a wrong segmentation or for keys that do not fit (r4b: dao read as 大哦).
- The base model must score whole sentences with its own tokenization; fed one character per token it underrates characters it normally writes inside a longer token (还 of 还有).
- `valid` is only comparable between runs with the same loss: it includes the KL term with `--teacher`, and the boundary cost of the letters with `--key-weight`.
- Before a full run of a new format: a few hundred steps, then `run_eval.py --limit 20` against the same 20 items of a known model.
- Remote `pgrep -f` / `pkill -f`: the pattern must not appear in your own ssh command line, or it matches itself (a wait loop never ends, pkill kills your ssh).
- dgx is unified memory: `--mem-frac 0.4` and `scripts/memguard.sh 12` beside training. Look at `nvidia-smi` on lab4090 / lab4090b first: other projects' training fills them.
