"""5B.8 check on M3: keys the user typed with one error (scripts/typo_stats.py cases), converted without and with the
typo readings: how often what was committed in the end is first / in the top 3. Prints counts only (D-019).

  cd DIR && ~/Projects/llm-ime/.venv/bin/python scripts/typo_eval.py [--dir LOGDIR] [--progress FILE] DAY ...

DIR: this code (git archive) with models -> ~/Projects/llm-ime/models. It loads two engines beside the service, so
it only runs while the user is not typing: when the newest typing log changes it stops (exit 3), keeping what it
has done in --progress (on M3, deleted when it is done), and the next run goes on from there.
"""

import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typo_stats  # noqa: E402
from ime import engine as engines  # noqa: E402

ENGINES = {"做前": dict(typo_penalty=0.0, swap_retries=1), "做后": {}}  # v02 without / with 5B.8


def newest_log(d):
    return max(os.stat(p).st_mtime for p in glob.glob(os.path.join(d, "*.jsonl")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=typo_stats.DEFAULT_DIR)
    ap.add_argument("--progress", default="/tmp/llmime-5b8/progress.jsonl")
    ap.add_argument("days", nargs="+")
    a = ap.parse_args()
    start = newest_log(a.dir)
    records = []
    for day in a.days:
        p = os.path.join(a.dir, day + ".jsonl")
        if os.path.exists(p):
            records += typo_stats.load(p)
    cases = list(dict.fromkeys(typo_stats.cases(records)))
    done = {}
    if os.path.exists(a.progress):
        for line in open(a.progress, encoding="utf-8"):
            r = json.loads(line)
            done[tuple(r["case"])] = r["cands"]
    todo = [c for c in cases if c not in done]
    if todo:
        engs = {name: engines.load_capped("v02", **kw) for name, kw in ENGINES.items()}
        with open(a.progress, "a", encoding="utf-8") as f:
            last_check = time.time()
            for c in todo:
                if time.time() - last_check >= 30:
                    last_check = time.time()
                    if newest_log(a.dir) != start:
                        print(f"typing again: stopped, {len(done)}/{len(cases)} done")
                        sys.exit(3)
                ctx, keys, _text, _kind = c
                done[c] = {name: e.convert(ctx, keys, k=5) for name, e in engs.items()}
                f.write(json.dumps({"case": c, "cands": done[c]}, ensure_ascii=False) + "\n")
                f.flush()
    hits = {name: collections.defaultdict(lambda: [0, 0, 0]) for name in ENGINES}  # kind -> [n, first, top 3]
    for c in cases:
        _ctx, _keys, text, kind = c
        for name in ENGINES:
            cands = done[c][name]
            for kd in (kind, "all"):
                h = hits[name][kd]
                h[0] += 1
                h[1] += bool(cands) and cands[0] == text
                h[2] += text in cands[:3]
    print(f"{len(cases)} 个错键串（{len(records)} 条记录）")
    for kd in ("extra", "miss", "near", "swap", "other", "all"):
        n = hits["做前"][kd][0]
        if n:
            print(f"{kd}: {n} 个  " + "  ".join(
                f"{name} 第 1 {hits[name][kd][1] / n:.0%} 前 3 {hits[name][kd][2] / n:.0%}" for name in ENGINES))
    os.remove(a.progress)


if __name__ == "__main__":
    main()
