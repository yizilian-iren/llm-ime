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
        from ime.llm import LLMEngine
        return LLMEngine(**kw)
    raise ValueError(f"unknown engine {name}")
