"""Build the eval set: corrected targets + the three input forms.

In:  data/private/user_selected.jsonl, data/private/user_corrections.json ({"fix": {index: text | null}})
     data/private/public_selected.jsonl (optional, same shape)
Out: data/private/eval.jsonl
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import pinyin  # noqa: E402

D = "data/private"


def load(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def main():
    fix = json.load(open(f"{D}/user_corrections.json"))["fix"]
    rows = []
    for i, r in enumerate(load(f"{D}/user_selected.jsonl")):
        if str(i) in fix:
            if fix[str(i)] is None:
                continue
            r["target"] = fix[str(i)]
        rows.append({**r, "src": "user"})
    rows += [{**r, "src": "public"} for r in load(f"{D}/public_selected.jsonl")]

    seen, out = set(), []
    for r in rows:
        if r["target"] in seen:
            continue
        seen.add(r["target"])
        rng = random.Random(r["id"])
        s = pinyin.syllables(r["target"])
        out.append({
            "id": r["id"], "src": r["src"], "context": r["context"], "target": r["target"],
            "sylls": s, "full": pinyin.full(s), "initials": pinyin.initials(s), "mixed": pinyin.mixed(s, rng),
        })
    with open(f"{D}/eval.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} items ({sum(r['src'] == 'user' for r in out)} user)")


if __name__ == "__main__":
    main()
