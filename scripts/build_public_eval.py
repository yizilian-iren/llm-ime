"""Public eval items: LCCC test utterances (chat, mostly no context) and held-out Wikipedia
sentences (context = the article text just before). → data/private/public_selected.jsonl
"""

import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import corpus  # noqa: E402

HAN_RUN = re.compile(r"[一-鿿]{6,24}")
N_EACH = 150


def item(ctx, target):
    key = hashlib.sha1((ctx + target).encode()).hexdigest()[:10]
    return {"id": "p" + key, "context": ctx[-64:], "target": target}


def main():
    rng = random.Random(0)
    chat = []
    for u in corpus.lccc(corpus.LCCC_TEST):
        for m in HAN_RUN.finditer(u):
            chat.append(item(u[: m.start()], m.group()))
    wiki = []
    for _, text in corpus.wiki_articles(held=True):
        text = text[:2000]
        for m in HAN_RUN.finditer(text):
            if m.start() > 20:  # skip titles and leads with no context
                wiki.append(item(text[: m.start()], m.group()))
        if len(wiki) > 20000:
            break
    out = rng.sample(chat, N_EACH) + rng.sample(wiki, N_EACH)
    os.makedirs("data/private", exist_ok=True)
    with open("data/private/public_selected.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(chat), len(wiki), len(out))


if __name__ == "__main__":
    main()
