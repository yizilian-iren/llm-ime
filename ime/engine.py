"""The interface every conversion engine implements, so the eval script can compare them."""

from typing import Protocol


class Engine(Protocol):
    name: str

    def convert(self, context: str, keys: str, k: int = 5) -> list[str]:
        """Candidates for the whole key string, best first.

        context: text left of the cursor (may be empty).
        keys: what the user typed, lowercase letters only (full pinyin, initials, or a mix).
        Called once per keystroke with the growing key string, so engines may cache by context / prefix.
        """
        ...


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
        cfg = dict(model="models/r2-4bit", mix="models/qwen3-0.6b-base-4bit", mix_weight=1.0, mix_weight_empty=0.5)
        return LLMEngine(**{**cfg, **kw})
    raise ValueError(f"unknown engine {name}")
