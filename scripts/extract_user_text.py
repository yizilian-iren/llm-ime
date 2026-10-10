"""Pull the user's own typed Chinese out of Claude Code transcripts into eval candidates.

Each candidate is a run of Han characters (4–24 chars) inside one message the user typed;
its context is the text before it in the same message (what an IME would see left of the cursor).
Output: data/private/user_candidates.jsonl
"""

import glob
import hashlib
import json
import os
import random
import re

SKIP_PREFIXES = ("Answers from your question cards", "你接替", "(A side question", "继续刚才", "[")
HAN_RUN = re.compile(r"[一-鿿]{4,24}")
HAN = re.compile(r"[一-鿿]")


def user_messages():
    seen = set()
    for f in sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl"))):
        for line in open(f, errors="ignore"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "user" or o.get("isMeta"):
                continue
            c = o.get("message", {}).get("content")
            if isinstance(c, list):
                c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
            if not isinstance(c, str) or c in seen:
                continue
            seen.add(c)
            c = c.strip()
            # Typed by the user, not pasted or written by an agent on their behalf.
            if c.startswith("<") or c.startswith(SKIP_PREFIXES) or "用户说" in c or "Agent Deck" in c:
                continue
            if "`" in c or len(c) > 400 or len(HAN.findall(c)) < 4:
                continue
            # Full-width punctuation and quote lines mean pasted agent text; the user types , ? half-width.
            if any(p in c for p in "，。：；「") or re.search(r"^>", c, re.M):
                continue
            yield c


def main():
    out = []
    for msg in user_messages():
        for m in HAN_RUN.finditer(msg):
            ctx = msg[: m.start()][-64:]
            key = hashlib.sha1((ctx + m.group()).encode()).hexdigest()[:10]
            out.append({"id": "u" + key, "context": ctx, "target": m.group()})
    rng = random.Random(0)
    rng.shuffle(out)
    os.makedirs("data/private", exist_ok=True)
    with open("data/private/user_candidates.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with_ctx = sum(1 for r in out if len(HAN.findall(r["context"])) >= 4)
    print(f"{len(out)} candidates, {with_ctx} with ≥4 Han chars of context")


if __name__ == "__main__":
    main()
