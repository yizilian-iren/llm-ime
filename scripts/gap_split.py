"""D-054: why one pass at beam 12 k 3 (scripts/passes_table.py) is 63 and the two passes' full search 72.5 (full keys,
P@1): the steps from the full search to the one-pass quick guess, each on its own, on the same items. Prints numbers only.

  python scripts/gap_split.py [--forms full,initials,mixed]

Variants (beam 12, the mixed-in model on), P@1 of the first candidate over three item sets (the first 100 items of the
eval set, as passes_table; the first 100 / 200 public ones, as quick_vs_full):
  full k5        engine.convert(k=5), the search of the second pass
  k3             k=3: the same, and no whole-word candidate (word_top acts for k > word_slot = 4 only)
  k3 no eos      k=3, eos=False: no extra step to score the end of the sentence
  chain          the quick guess typed key by key (a Session's `_quick`: whole search under 10 keys, from the last best
                 for 10 or more, no end score): what one pass shows
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ime import engine as engines  # noqa: E402
from quick_split import session  # noqa: E402
from run_eval import keys_of  # noqa: E402

VARIANTS = ["full k5", "k3", "k3 no eos", "chain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default="full,initials,mixed")
    a = ap.parse_args()
    eng = engines.load_capped("v02")
    allitems = [json.loads(line) for line in open("data/private/eval.jsonl")]
    pub = [r for r in allitems if r["src"] == "public"]
    sets = {"eval first 100": allitems[:100], "public first 100": pub[:100], "public first 200": pub[:200]}
    todo = {r["id"]: r for s in sets.values() for r in s}
    eng.convert("", "nihao", k=1)
    ok = collections.defaultdict(dict)  # (form, variant) -> id -> bool
    for n, r in enumerate(todo.values()):
        for form in a.forms.split(","):
            keys = keys_of(r, form)
            if keys is None:
                continue
            ctx, tgt = r["context"], r["target"]
            s = session(eng, beam=12)
            out = {"full k5": eng.convert(ctx, keys, k=5), "k3": eng.convert(ctx, keys, k=3),
                   "k3 no eos": eng.convert(ctx, keys, k=3, eos=False)}
            for j in range(1, len(keys) + 1):
                chain = s._quick(ctx, keys[:j])
            out["chain"] = chain
            for v in VARIANTS:
                ok[form, v][r["id"]] = bool(out[v]) and out[v][0] == tgt
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(todo)}", file=sys.stderr)
    for form in a.forms.split(","):
        print(f"## {form}")
        for name, items in sets.items():
            ids = [r["id"] for r in items if r["id"] in ok[form, "chain"]]
            print(f"{name:17s} n={len(ids)}  " + "  ".join(
                f"{v} {100 * sum(ok[form, v][i] for i in ids) / len(ids):.1f}" for v in VARIANTS))


if __name__ == "__main__":
    main()
