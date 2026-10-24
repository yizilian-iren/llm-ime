"""Prompt format shared by training and the engine. Three orders (a model's `ime_format` file names its own;
an empty file is the first):
  ctx_first:  context SEP1 k e y s SEP2 target EOS
  keys_first: SEP1 k e y s SEP2 context target EOS   (target continues the context directly, as plain text)
  interleave: context SEP1 h a i 还 y o u 有 ... EOS  (each character right after its own keys, one token per
              character; no later key is visible, so a hypothesis's cache stays valid as keys are added; r4b)

Private-use separators (as in Zenzai) can't collide with real text. Keys are spaced so every letter is
its own token whatever the BPE merges would have done.
An order may name its separators after a `+` (`ctx_first+quad`, 7B.4): Qwen3.5's vocab has no token for the
private-use characters (3 byte tokens each, one more forward per key), so its models use two of its own special
tokens, one token each and never merged with the text beside them. `open` has no end separator: the target follows
the last key directly (a key is " x", never a target's first token), so the logits after the last key are the first
target step's and a key costs one forward of a recurrent model, not two (the state before SEP2 is kept apart).
"""

SEP1, SEP2 = "\uee00", "\uee01"
SEPS = {"pua": (SEP1, SEP2), "quad": ("<|quad_start|>", "<|quad_end|>"), "open": ("<|quad_start|>", "")}
CTX_CHARS = 64


ORDERS = ("ctx_first", "keys_first", "interleave")


def base(order: str) -> str:
    """The order without its separators: ctx_first+quad → ctx_first."""
    return order.split("+")[0]


def seps(order: str = "ctx_first") -> tuple[str, str]:
    return SEPS[order.split("+")[1] if "+" in order else "pua"]


def parse_order(order: str) -> str:
    """argparse type for --order: an order of ORDERS, with or without +<separators> of SEPS."""
    b, _, s = order.partition("+")
    if b not in ORDERS or (s and s not in SEPS):
        raise ValueError(order)
    return order


def keys_part(keys: str, order: str = "ctx_first") -> str:
    s1, s2 = seps(order)
    return f"{s1}{' '.join(keys)}{s2}"


def prompt(context: str, keys: str, order: str = "ctx_first") -> str:
    if base(order) == "keys_first":
        return keys_part(keys, order) + context[-CTX_CHARS:]
    return context[-CTX_CHARS:] + keys_part(keys, order)


def keys_in(text: str) -> str | None:
    """The spaced keys between the last pair of separators of any set in a decoded prompt, or None."""
    for s1, s2 in SEPS.values():
        a, b = text.rfind(s1), (text.rfind(s2) if s2 else len(text))
        if 0 <= a < b:
            return text[a + len(s1) : b]
    return None


def interleave(encode, context: str, pieces: list[str], chars: str, order: str = "interleave") -> tuple[list[int], int]:
    """interleave order as token ids, built piece by piece so training and the engine tokenize alike:
    (ids, where the first key starts). pieces: each character's keys; encode: str -> token ids."""
    head = (encode(context[-CTX_CHARS:]) if context else []) + encode(seps(order)[0])
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
