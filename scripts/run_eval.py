"""Run an engine over the eval set: P@1 / P@5 by bucket, plus per-keystroke latency.

  python scripts/run_eval.py --engine ngram [--forms full,initials,mixed] [--limit N] [--latency 40]

Forms full_swap / mixed_swap: the full / mixed keys with two adjacent letters of one typed syllable exchanged
(one per item, fixed by its id; items with no such pair are left out), for the swap tolerance (5B.1).

Accuracy: one call per item with the whole key string.
Latency: on the first --latency items (initials form), type key by key and time every call.
Per-item outputs go to data/private/results/<engine>-<tag>.jsonl (they contain the user's text).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime import pinyin  # noqa: E402

HAN = re.compile(r"[一-鿿]")
EXAMPLES = [  # the user's own complaints, always reported one by one
    ("", "还有很多打字时根本不看上下文就这样我很烦"),
    ("这个方案你觉得怎么样", "还有别的办法呢"),
    ("现在是不是没问题", "呢"),
]


def swapped(r, form):
    """r's full / mixed keys with one adjacent pair inside a typed syllable exchanged, or None if none can be."""
    parts = r["sylls"] if form == "full" else pinyin.mixed_parts(r["sylls"], random.Random(r["id"]))  # as build_eval
    assert "".join(parts) == r[form], (r["id"], form)
    pairs = [(i, j) for i, s in enumerate(parts) for j in range(len(s) - 1) if s[j] != s[j + 1]]
    if not pairs:
        return None
    i, j = random.Random(f"{r['id']}-{form}").choice(pairs)
    s = parts[i]
    return "".join(parts[:i]) + s[:j] + s[j + 1] + s[j] + s[j + 2:] + "".join(parts[i + 1:])


def keys_of(r, form):
    return swapped(r, form.removesuffix("_swap")) if form.endswith("_swap") else r[form]


def ctx_bucket(ctx):
    n = len(HAN.findall(ctx))
    return "ctx=0" if n < 4 else ("ctx<16" if n < 16 else "ctx≥16")


def len_bucket(t):
    return "len≤6" if len(t) <= 6 else ("len7-10" if len(t) <= 10 else "len≥11")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--forms", default="full,initials,mixed")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--latency", type=int, default=40)
    ap.add_argument("--src", default="", help="only items from this source (user / public)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--opt", action="append", default=[], help="engine option key=value")
    ap.add_argument("--words", default="", help="user words to rerank with (5A.3): keys<TAB>word[<TAB>count] lines")
    a = ap.parse_args()

    opts = {}
    for o in a.opt:
        k, v = o.split("=", 1)
        try:
            opts[k] = int(v) if v.isdigit() else float(v)
        except ValueError:
            opts[k] = v
    eng = engines.load(a.engine, **opts)
    words = None
    if a.words:
        from ime.userwords import UserWords
        words = UserWords(eng, path="")
        print(f"{words.import_file(a.words)} user words", file=sys.stderr)
    items = [json.loads(line) for line in open("data/private/eval.jsonl")]
    if a.src:
        items = [r for r in items if r["src"] == a.src]
    if a.limit:
        items = items[: a.limit]
    forms = a.forms.split(",")

    hits = defaultdict(lambda: [0, 0, 0])  # bucket -> [n, p1, p5]
    plain_hits = [0, 0, 0]  # with --words: n, P@1 without them, first candidate changed by them
    out = []
    t0 = time.time()
    for i, r in enumerate(items):
        for form in forms:
            keys = keys_of(r, form)
            if keys is None:
                continue
            cands = eng.convert(r["context"], keys, k=5)
            if words:
                plain = cands
                cands = words.rerank(r["context"], keys, cands)
                plain_hits[0] += 1
                plain_hits[1] += bool(plain) and plain[0] == r["target"]
                plain_hits[2] += bool(plain) and bool(cands) and plain[0] != cands[0]
            p1 = bool(cands) and cands[0] == r["target"]
            p5 = r["target"] in cands[:5]
            for b in ("all", f"{form}", f"{form} {ctx_bucket(r['context'])}", f"{form} {len_bucket(r['target'])}",
                      f"{form} src={r['src']}"):
                h = hits[b]
                h[0] += 1
                h[1] += p1
                h[2] += p5
            out.append({"id": r["id"], "form": form, "keys": keys, "target": r["target"], "cands": cands, "p1": p1})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)}  {time.time() - t0:.0f}s", file=sys.stderr)

    lat = []
    for r in items[: a.latency]:
        keys = r["initials"]
        for j in range(1, len(keys) + 1):
            s = time.perf_counter()
            eng.convert(r["context"], keys[:j], k=5)
            lat.append((time.perf_counter() - s) * 1000)

    os.makedirs("data/private/results", exist_ok=True)
    tag = f"{a.engine}{'-' + a.tag if a.tag else ''}"
    if a.limit or a.src:  # partial runs never overwrite a full run's results
        tag += f"-part{a.limit or ''}{a.src}"
    with open(f"data/private/results/{tag}.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"\n## {tag}  ({len(items)} items, {time.time() - t0:.0f}s)\n")
    print("| bucket | n | P@1 | P@5 |\n|---|---:|---:|---:|")
    for b in sorted(hits, key=lambda b: (b != "all", b.split()[0], b)):
        n, p1, p5 = hits[b]
        print(f"| {b} | {n} | {100 * p1 / n:.1f} | {100 * p5 / n:.1f} |")
    if words:
        n, p1, ch = plain_hits
        print(f"\nwithout the user words: P@1 {100 * p1 / n:.1f}; their first candidate changed on {ch} of {n}")
    if lat:
        print(f"\nper-keystroke latency (initials, {len(lat)} calls): p50 {pct(lat, .5):.1f} ms, p95 {pct(lat, .95):.1f} ms")

    print("\nexamples:")
    for ctx, tgt in EXAMPLES:
        from ime import pinyin
        s = pinyin.syllables(tgt)
        for form, keys in (("initials", pinyin.initials(s)), ("full", pinyin.full(s))):
            print(f"  [{ctx}] {keys} → {eng.convert(ctx, keys, k=5)}")


if __name__ == "__main__":
    main()
