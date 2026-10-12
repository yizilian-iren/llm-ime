"""Prompt format shared by training and the engine. Three orders (a model's `ime_format` file names its own;
an empty file is the first):
  ctx_first:  context SEP1 k e y s SEP2 target EOS
  keys_first: SEP1 k e y s SEP2 context target EOS   (target continues the context directly, as plain text)
  interleave: context SEP1 h a i 还 y o u 有 ... EOS  (each character right after its own keys, one token per
              character; no later key is visible, so a hypothesis's cache stays valid as keys are added; r4b)

Private-use separators (as in Zenzai) can't collide with real text. Keys are spaced so every letter is
its own token whatever the BPE merges would have done.
"""

SEP1, SEP2 = "", ""
CTX_CHARS = 64


ORDERS = ("ctx_first", "keys_first", "interleave")


def keys_part(keys: str) -> str:
    return f"{SEP1}{' '.join(keys)}{SEP2}"


def prompt(context: str, keys: str, order: str = "ctx_first") -> str:
    if order == "keys_first":
        return keys_part(keys) + context[-CTX_CHARS:]
    return context[-CTX_CHARS:] + keys_part(keys)


def interleave(encode, context: str, pieces: list[str], chars: str) -> tuple[list[int], int]:
    """interleave order as token ids, built piece by piece so training and the engine tokenize alike:
    (ids, where the first key starts). pieces: each character's keys; encode: str -> token ids."""
    head = (encode(context[-CTX_CHARS:]) if context else []) + encode(SEP1)
    body = []
    for piece, ch in zip(pieces, chars):
        body += [encode(c)[0] for c in piece] + encode(ch)
    return head + body, len(head)


def model_order(model_dir: str) -> str | None:
    """The order a fine-tuned model was trained with, or None for a base model (no `ime_format` file)."""
    import os

    path = os.path.join(model_dir, "ime_format")
    if not os.path.exists(path):
        return None
    return open(path).read().strip() or "ctx_first"
