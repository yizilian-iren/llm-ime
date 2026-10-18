"""The interface every conversion engine implements, so the eval script can compare them."""

import os
from typing import Protocol

V02_BASE = "models/r2-4bit"  # the fine-tuned model engine v02 is built on: the base the weekly retrain trains on
PERSONAL = "models/personal-4bit"  # the weekly retrain's model (6A.2): a LoRA on the user's commits, merged


class Engine(Protocol):
    name: str

    def convert(self, context: str, keys: str, k: int = 5) -> list[str]:
        """Candidates for the whole key string, best first.

        context: text left of the cursor (may be empty).
        keys: what the user typed, lowercase letters only (full pinyin, initials, or a mix).
        Called once per keystroke with the growing key string, so engines may cache by context / prefix.
        """
        ...


def base_model(name: str = "v02") -> str:
    """The base the weekly retrain (scripts/retrain.py) trains on: the one deployed with this code."""
    assert name == "v02", name
    return V02_BASE


def personal_or(base: str) -> str:
    """The personal model when it was trained on `base` (its `base` file names it), else `base`: a new base makes
    the old personal model void until the next retrain; deleting the personal model goes back to the base."""
    try:
        with open(os.path.join(PERSONAL, "base")) as f:
            if f.read().strip() == os.path.basename(base):
                return PERSONAL
    except OSError:
        pass
    return base


def load(name: str, **kw) -> Engine:
    if name == "ngram":
        from ime.ngram import NgramEngine
        return NgramEngine(**kw)
    if name == "llm":
        from ime import fmt
        if fmt.model_order(kw.get("model", "")) == "interleave":  # r4b
            from ime.interleave import InterleaveEngine
            return InterleaveEngine(**kw)
        from ime.llm import LLMEngine
        return LLMEngine(**kw)
    if name == "v02":  # v0.2's best: fine-tuned r2 with the base model mixed in (2A.3)
        from ime.llm import LLMEngine
        cfg = dict(model=personal_or(V02_BASE), mix="models/qwen3-0.6b-base-4bit", mix_weight=1.0, mix_weight_empty=0.5,
                   split_penalty=9.0,  # ma read as 命啊 / 没啊 instead of 吗 (user on M3, v0.4)
                   swap_penalty=3.0,  # zhognguo read as 中国 (5B.1)
                   alt_bonus=6.0,  # en read as 嗯: enne 嗯呢, enen 嗯嗯 first, en 嗯 in the top 3 (5B.4)
                   single_weight=0.6,  # one syllable typed: common characters first, wo 我 not 沃 (5B.5)
                   english=True,  # github, cpu, tracker as typed first when not pinyin (5B.9)
                   typo_penalty=4.0, swap_retries=2)  # yiijingshi 已经是, kanyuix 看一下, shibnush 是不是 (5B.8)
        return LLMEngine(**{**cfg, **kw})
    raise ValueError(f"unknown engine {name}")
