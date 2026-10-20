"""The user's commits replayed through engine variants (6B.5), on M3 where the typing log is (never copied off, D-019):
counts, and with --show the misses one by one.

  # on M3: this branch's code in a folder of its own, models from the repo there
  git archive HEAD | ssh m3 'rm -rf /tmp/llmime-6b5 && mkdir /tmp/llmime-6b5 && tar -x -C /tmp/llmime-6b5'
  ssh m3 'cd ~/Projects/llm-ime && nice .venv/bin/python /tmp/llmime-6b5/scripts/word_replay.py \
      --model models/personal-20261018-4bit --variants base,word3,first3'

It runs only while the user is away: before each item it waits until M3 has had no key or mouse input for 2 min.
A variant may carry engine options: base@word_top=30000,word_floor=2.
Items: commits of Han characters by space / digit / enter whose text is one of the candidates shown for all the keys.
"word misses": the first candidate was not committed and is, unlike the commit, not a jieba word (先在 for 现在) or joined
by jieba to the context's last character (修 + 成与); "controls": --controls commits whose first candidate was committed
(a fixed sample). No user word list (its rerank is left out), so base is the model alone. Prints per variant how many
word misses come first and how many controls stay first.
"""

import argparse
import glob
import json
import os
import random
import re
import subprocess
import sys
import time

import jieba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from word_ctx_eval import dict_words, patch  # noqa: E402

from ime import engine as engines  # noqa: E402
from ime import llm  # noqa: E402

HAN = re.compile(r"^[一-鿿]+$")
LOG = os.path.expanduser("~/Library/Application Support/llm-ime/log")


def wait_idle(idle=120.0):
    """Until the user has not touched M3's keyboard or mouse for `idle` s (IOHIDSystem's HIDIdleTime): the replay must
    not slow the input method while they type."""
    while True:
        out = subprocess.run(["ioreg", "-c", "IOHIDSystem"], capture_output=True, text=True).stdout
        m = re.search(r'"HIDIdleTime" = (\d+)', out)
        if not m or int(m.group(1)) / 1e9 >= idle:
            return
        time.sleep(30)


def convert(eng, c, k):
    wait_idle()
    return eng.convert(c, k, k=5)


def cuts(text):
    at, out = 0, set()
    for w in jieba.lcut(text):
        at += len(w)
        out.add(at)
    return out


def items(controls):
    misses, ok = [], []
    for p in sorted(glob.glob(os.path.join(LOG, "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            text, cands, ctx = r.get("text") or "", r.get("cands") or [], r.get("context") or ""
            if r.get("how") not in ("space", "digit", "enter") or not HAN.match(text) or text not in cands:
                continue
            row = (ctx, r["keys"], text)
            if cands[0] == text:
                ok.append(row)
                continue
            first = cands[0]
            tail = re.search(r"[一-鿿]{1,16}$", ctx)
            split = len(first) == len(text) >= 2 and jieba.lcut(text) == [text] and len(jieba.lcut(first)) > 1
            cross = tail and len(tail.group()) not in cuts(tail.group() + first) \
                and len(tail.group()) in cuts(tail.group() + text)
            if split or cross:
                misses.append(row)
    return misses, random.Random(7).sample(ok, min(controls, len(ok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/personal-20261018-4bit")
    ap.add_argument("--variants", default="base,word3,first3")
    ap.add_argument("--controls", type=int, default=300)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    jieba.setLogLevel(60)
    common, pairs = dict_words()
    misses, ok = items(a.controls)
    whole = llm._words(100000)  # 6B.7: the commit is one word read from the whole keys, or a piece of something longer
    n_whole = sum(any(w == t for w, _s in whole.get(k, ())) for _c, k, t in misses)
    print(f"word misses n={len(misses)}: the keys are the whole pinyin of the commit, a word of ime/words.tsv: {n_whole}; "
          f"longer than a word: {len(misses) - n_whole}", flush=True)
    firsts, in5 = {}, {}
    for v in a.variants.split(","):
        wait_idle()
        name, _, opt = v.partition("@")  # base@word_top=30000,word_floor=2: engine options after @ (6B.7)
        eng = engines.load("v02", model=a.model, **{k: float(x) for k, x in (o.split("=") for o in opt.split(",") if o)})
        patch(eng, name, common, pairs)
        got = [convert(eng, c, k) for c, k, _t in misses]
        kept = sum(bool(x) and x[0] == t for (c, k, t), x in zip(ok, (convert(eng, c, k) for c, k, _t in ok)))
        fixed = sum(bool(x) and x[0] == t for (_c, _k, t), x in zip(misses, got))
        top5 = sum(t in x for (_c, _k, t), x in zip(misses, got))
        print(f"{v:8} word misses n={len(misses)}: first {fixed}, top 5 {top5} | controls n={len(ok)}: "
              f"still first {kept}", flush=True)
        firsts[v] = [x[0] if x else "" for x in got]
        in5[v] = [t in x for (_c, _k, t), x in zip(misses, got)]
        if len(in5) > 1:  # counts against the first variant: who got into the top 5, who dropped out, by kind (6B.7)
            base = in5[next(iter(in5))]
            for kind, pick in (("whole word", lambda k, t: any(w == t for w, _s in whole.get(k, ()))),
                               ("longer", lambda k, t: not any(w == t for w, _s in whole.get(k, ())))):
                idx = [i for i, (_c, k, t) in enumerate(misses) if pick(k, t)]
                print(f"         {kind} n={len(idx)}: top 5 in {sum(in5[v][i] for i in idx)} (first variant "
                      f"{sum(base[i] for i in idx)}), gained {sum(in5[v][i] and not base[i] for i in idx)}, "
                      f"lost {sum(base[i] and not in5[v][i] for i in idx)}", flush=True)
        del eng
    if a.show:
        for i, (c, k, t) in enumerate(misses):
            print(f"  [{c[-10:]}] {k} -> {t}: " + " ".join(f"{v}:{firsts[v][i]}" for v in firsts))


if __name__ == "__main__":
    main()
