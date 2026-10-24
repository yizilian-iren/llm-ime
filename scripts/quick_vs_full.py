"""7D.1: where the full answer's first candidate is worse than the quick guess's, on the same sentences.

  python scripts/quick_vs_full.py [--items 200] [--forms full,initials,mixed] [--examples 5] [--words-n 50]

Public eval items only (their sentences may be printed). Each key string is typed one key at a time through a Session's
`_quick` (its own last best, as the quick guess starts from it for 10 keys or more; no full search between keys), the
last key's quick guess against `_convert(k=5)`, the full search. Four cells by whether each first candidate is the
target. Then, for "quick right, full wrong", which part of the full search is to blame: the full search run again
with that part off, in the order word (word_top=0), typo rows (no typo_penalty), end score (eos=False), beam (beam=3, as
quick); the first that makes the first candidate the target is named, else "other". With --words-n, the same cells with
a list of that many test user words (taken from public items beyond --items, two syllables each, count 3; never M3's
words.json): the quick guess gets `words.whole`, the full answer `words.rerank`, as in Session. Prints numbers only,
besides the public examples.
"""

import argparse
import collections
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ime import engine as engines  # noqa: E402
from ime.userwords import UserWords  # noqa: E402
from quick_split import session  # noqa: E402
from run_eval import keys_of  # noqa: E402


def cell(q, f):
    return ("both right" if f else "quick right, full wrong") if q else ("quick wrong, full right" if f else "both wrong")


def whole_check(eng, words, src, n_examples):
    """7D.1 step 4: the keys are a learned word's whole keys. Quick puts the word first without scoring (`whole`), full
    scores it with its bonus (`rerank`): how often the first candidates differ, with the sentence before as context
    and without."""
    cnt, shown = collections.Counter(), []
    for r in src:
        keys, word = "".join(r["sylls"][:2]), r["target"][:2]
        for label, ctx in (("with context", r["context"]), ("no context", "")):
            s = session(eng)
            for j in range(1, len(keys) + 1):
                quick = s._quick(ctx, keys[:j])
            qw = words.whole(keys, quick)
            plain = s._convert(ctx, keys, k=5)
            fw = words.rerank(ctx, keys, plain)
            q1, f1, p1 = qw[0] == word, fw[0] == word, plain[0] == word
            cnt[label, "n"] += 1
            cnt[label, "quick first is the word"] += q1
            cnt[label, "full first is the word"] += f1
            cnt[label, "full without the words first is the word"] += p1
            cnt[label, "quick first, full not"] += q1 and not f1
            cnt[label, "full first, quick not"] += f1 and not q1
            if q1 and not f1 and len(shown) < n_examples and label == "with context":
                shown.append((ctx[-10:], keys, word, "quick", qw[0], "full", fw[0], "plain full", plain[0]))
    print("## learned word = the whole keys (step 4)")
    for label in ("with context", "no context"):
        print(label, "  ".join(f"{k} {v}" for (l, k), v in cnt.items() if l == label))
    for e in shown:
        print("  ", *e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--forms", default="full,initials,mixed")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--words-n", type=int, default=50)
    ap.add_argument("--whole-only", action="store_true", help="only the learned-word check (step 4), not the cells")
    a = ap.parse_args()
    if a.whole_only:
        a.items = 0
    eng = engines.load("v02")
    pub = [r for r in map(json.loads, open("data/private/eval.jsonl")) if r["src"] == "public"]
    items, rest = pub[: a.items], pub[a.items:]
    words = None
    if a.words_n:
        path = os.path.join(tempfile.mkdtemp(), "words.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for r in rest[: a.words_n]:
                f.write(f"{''.join(r['sylls'][:2])}\t{r['target'][:2]}\t3\n")
        words = UserWords(eng, path="")
        print(f"{words.import_file(path)} test user words", file=sys.stderr)
    eng.convert("", "nihao", k=1)
    typos, typo_penalty = eng.typos, eng.typo_penalty
    if words:
        if a.whole_only:  # words last used 0 / 30 / 60 / 90 days ago: their bonus halves every 30 days
            import time
            for age in (0, 30, 60, 90):
                aged = UserWords(eng, path="")
                for r in rest[: a.words_n]:
                    aged.add("".join(r["sylls"][:2]), r["target"][:2], 3, now=time.time() - age * 86400)
                print(f"### last used {age} days ago")
                whole_check(eng, aged, rest[: a.words_n], a.examples)
            return
        whole_check(eng, words, rest[: a.words_n], a.examples)
    cells = collections.Counter()
    why = collections.Counter()
    shown = []
    wcells, flipped = collections.Counter(), collections.Counter()
    for i, r in enumerate(items):
        for form in a.forms.split(","):
            keys, ctx, tgt = keys_of(r, form), r["context"], r["target"]
            if keys is None:
                continue
            s = session(eng)
            for j in range(1, len(keys) + 1):
                quick = s._quick(ctx, keys[:j])
            full = s._convert(ctx, keys, k=5)
            q_ok, f_ok = bool(quick) and quick[0] == tgt, bool(full) and full[0] == tgt
            cells[form, cell(q_ok, f_ok)] += 1
            if q_ok and not f_ok:
                cause = "other"
                word_top = eng.word_top
                ablate = [("word", lambda: setattr(eng, "word_top", 0)),
                          ("typo rows", lambda: (setattr(eng, "typo_penalty", 0.0), setattr(eng, "typos", {}))),
                          ("end score", None), ("beam 3", None)]
                for name, fn in ablate:
                    if name == "word":
                        eng.word_top = 0
                        c = s._convert(ctx, keys, k=5)
                        eng.word_top = word_top
                    elif name == "typo rows":
                        eng.typo_penalty, eng.typos = 0.0, {}
                        c = s._convert(ctx, keys, k=5)
                        eng.typo_penalty, eng.typos = typo_penalty, typos
                    elif name == "end score":
                        c = s._convert(ctx, keys, k=5, eos=False)
                    else:
                        c = s._convert(ctx, keys, k=5, beam=3)
                    if c and c[0] == tgt:
                        cause = name
                        break
                why[form, cause] += 1
                if len(shown) < a.examples and form == "full":
                    shown.append((ctx[-12:], keys, tgt, quick[0], full[0], cause))
            if words:
                qw, fw = words.whole(keys, quick), words.rerank(ctx, keys, full)
                qw_ok, fw_ok = bool(qw) and qw[0] == tgt, bool(fw) and fw[0] == tgt
                wcells[form, cell(qw_ok, fw_ok)] += 1
                flipped[form, "rerank right -> wrong"] += f_ok and not fw_ok
                flipped[form, "rerank wrong -> right"] += fw_ok and not f_ok
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)}", file=sys.stderr)
    for title, c in (("no user words", cells), (f"{a.words_n} test user words", wcells)):
        if not c:
            continue
        print(f"## {title}, {len(items)} public items")
        for form in a.forms.split(","):
            n = sum(v for (f, _), v in c.items() if f == form)
            print(f"{form:8s} n={n}  " + "  |  ".join(
                f"{k} {c[form, k]}" for k in ("both right", "quick right, full wrong", "quick wrong, full right", "both wrong")))
    print("## quick right, full wrong: which part of the full search")
    for form in a.forms.split(","):
        print(f"{form:8s} " + "  ".join(f"{c} {n}" for (f, c), n in sorted(why.items()) if f == form))
    if words:
        print("## the rerank by the test user words, full answer's first")
        for form in a.forms.split(","):
            print(f"{form:8s} " + "  ".join(f"{c} {n}" for (f, c), n in sorted(flipped.items()) if f == form))
    print("## examples (full form)")
    for e in shown:
        print("  ", *e)


if __name__ == "__main__":
    main()
