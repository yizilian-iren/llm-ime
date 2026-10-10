"""Prompt format shared by training and the engine: context SEP1 k e y s SEP2 target EOS.

Private-use separators (as in Zenzai) can't collide with real text. Keys are spaced so every letter is
its own token whatever the BPE merges would have done.
"""

SEP1, SEP2 = "", ""
CTX_CHARS = 64


def prompt(context: str, keys: str) -> str:
    return f"{context[-CTX_CHARS:]}{SEP1}{' '.join(keys)}{SEP2}"
