"""Prompt format shared by training and the engine. Two orders (a model's `ime_format` file names its own;
an empty file is the first):
  ctx_first:  context SEP1 k e y s SEP2 target EOS
  keys_first: SEP1 k e y s SEP2 context target EOS   (target continues the context directly, as plain text)

Private-use separators (as in Zenzai) can't collide with real text. Keys are spaced so every letter is
its own token whatever the BPE merges would have done.
"""

SEP1, SEP2 = "", ""
CTX_CHARS = 64


ORDERS = ("ctx_first", "keys_first")


def keys_part(keys: str) -> str:
    return f"{SEP1}{' '.join(keys)}{SEP2}"


def prompt(context: str, keys: str, order: str = "ctx_first") -> str:
    if order == "keys_first":
        return keys_part(keys) + context[-CTX_CHARS:]
    return context[-CTX_CHARS:] + keys_part(keys)


def model_order(model_dir: str) -> str | None:
    """The order a fine-tuned model was trained with, or None for a base model (no `ime_format` file)."""
    import os

    path = os.path.join(model_dir, "ime_format")
    if not os.path.exists(path):
        return None
    return open(path).read().strip() or "ctx_first"
