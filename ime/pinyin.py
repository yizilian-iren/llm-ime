"""Pinyin helpers: text → syllables, and syllables → the keystrokes a user might type."""

import random
import re

from pypinyin import Style, lazy_pinyin, pinyin

HAN = re.compile(r"[一-鿿]")


def syllables(text: str) -> list[str]:
    """Toneless pinyin per character, ü written as v (what people type)."""
    return [s.replace("ü", "v") for s in lazy_pinyin(text, style=Style.NORMAL, v_to_u=False)]


def char_readings(ch: str) -> list[str]:
    """Every toneless reading of one character (polyphones give several)."""
    rs = pinyin(ch, style=Style.NORMAL, heteronym=True, v_to_u=False)[0]
    return sorted({r.replace("ü", "v") for r in rs})


def full(sylls: list[str]) -> str:
    return "".join(sylls)


def initials(sylls: list[str]) -> str:
    return "".join(s[0] for s in sylls)


def mixed(sylls: list[str], rng: random.Random) -> str:
    return "".join(mixed_parts(sylls, rng))


def mixed_parts(sylls: list[str], rng: random.Random) -> list[str]:
    """Some syllables typed in full, some as the first letter, some half-typed: the keys per character."""
    out = []
    for s in sylls:
        r = rng.random()
        if r < 0.4 or len(s) == 1:
            out.append(s)
        elif r < 0.8:
            out.append(s[0])
        else:
            out.append(s[: rng.randint(1, len(s) - 1)] if len(s) > 2 else s[0])
    return out
