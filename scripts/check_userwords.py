"""Checks the user's words (5A.3) with the real engine, on a word list in a temporary file:

  .venv/bin/python scripts/check_userwords.py [--engine v02]

Each case: the rank of the user's word before, then after one pick of it (as the input method reports a pick:
keys, what was committed, the engine's first candidate then). Exits 1 when a case misses its goal.
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime.userwords import UserWords  # noqa: E402

# (picked: context, keys, text; or a word list [(keys, word, count)]), (typed later: context, keys), the word, the rank
# it must reach (1 = first) or the word it must come before
CASES = [
    (("我最近在玩", "mingchao", "鸣潮"), ("我最近在玩", "mingchao"), "鸣潮", 1),
    (("我最近在玩", "mingchao", "鸣潮"), ("", "wozuijinzaiwanmingchao"), "我最近在玩鸣潮", 1),
    (("还有给我讲解一下", "liji", "力积"), ("能不能讲讲", "liji"), "力积", 5),
    (("我", "baba", "爸爸"), ("我", "baba"), "爸爸", "粑粑"),  # before 粑粑
    (("这些怎么", "zhexiezenmede", "这些怎么得"), ("你怎么", "zenmede"), "怎么得", 2),
    # 先 picked alone 5 times, 现在 3 times: xianz is still 现在, not 先在 (the best's own word counts)
    ([("xian", "先", 5), ("xianz", "现在", 3)], ("", "xianz"), "现在", 1),
    # 5A.5, given up and typed again one character at a time (a fourth item: the first candidate the input method
    # reports then): Esc on mingchao before any answer, then 鸣, 潮
    (("", "mingchao", "鸣潮", ""), ("我最近在玩", "mingchao"), "鸣潮", 1),
    # the sentence deleted back to wozuijinzaiwan and committed, then 鸣, 潮: only the differing 鸣潮 is learned
    (("", "wozuijinzaiwanmingchao", "我最近在玩鸣潮", "我最近在玩明朝"), ("我最近在玩", "mingchao"), "鸣潮", 1),
]


def rank(cands, w):
    return cands.index(w) + 1 if w in cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="v02")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    eng = engines.load(a.engine)
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "words.json")
        for picked, (ctx, keys), word, goal in CASES:
            words = UserWords(eng, path)
            words.forget()
            before = eng.convert(ctx, keys, k=5)
            if isinstance(picked, list):  # a word list as it was: (keys, word, count)
                for k, w, n in picked:
                    words.add(k, w, n)
            else:
                pctx, pkeys, ptext, *pfirst = picked
                words.learn(pkeys, ptext, pfirst[0] if pfirst else eng.convert(pctx, pkeys, k=5)[0])
            t = time.perf_counter()
            after = words.rerank(ctx, keys, eng.convert(ctx, keys, k=5))
            ms = (time.perf_counter() - t) * 1000
            r = rank(after, word)
            ok = r is not None and (r <= goal if isinstance(goal, int) else rank(after, goal) is None or r < rank(after, goal))
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} {ctx}+{keys}: {word} {rank(before, word)} -> {r} (goal {goal}); "
                  f"learned {words.words}; {after[:5]} ({ms:.0f} ms with the search)")
        words = UserWords(eng, path)
        words.forget()
        words.learn("mingchao", "鸣潮的", "")  # 5A.5: typed on past the keys given up on: does not read them
        words.learn("wozuijinzaiwanmingchao", "我最近在玩鸣潮", "")  # no first, more than four: says too little
        ok = not words.words
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} texts that do not read the keys, or long with no first, not learned: {words.words}")
        words.learn("mingchao", "鸣潮", "明朝")
        words.forget()
        ok = not os.path.exists(path) and not UserWords(eng, path).words
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} forget empties the list")
    print("userwords: all passed" if not failed else f"userwords: {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
