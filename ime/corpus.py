"""Public corpora: LCCC (everyday conversation) and a Chinese Wikipedia shard.

Wikipedia articles whose id hashes to 0 mod 50 are held out for the eval set and never counted.
"""

import gzip
import json
import re
import zlib

LCCC_TRAIN = "data/corpus/lccc/lccc_base_train.jsonl.gz"
LCCC_TEST = "data/corpus/lccc/lccc_base_test.jsonl.gz"
WIKI = "data/corpus/wiki/0.parquet"

LINE_SPLIT = re.compile(r"[\n。！？!?；;]+")


def held_out(article_id: str) -> bool:
    return zlib.crc32(article_id.encode()) % 50 == 0


def lccc(path=LCCC_TRAIN):
    """Utterances, spaces removed (LCCC ships pre-tokenized)."""
    with gzip.open(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            for u in d["dialog"] if isinstance(d, dict) else d:
                yield u.replace(" ", "")


def wiki_articles(held=False):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(WIKI)
    for batch in pf.iter_batches(columns=["id", "text"], batch_size=2000):
        d = batch.to_pydict()
        for aid, text in zip(d["id"], d["text"]):
            if held_out(str(aid)) == held:
                yield str(aid), text


def wiki_with_prefix(max_chars=60_000_000, prefix_chars=64):
    """(article text just before the sentence, sentence): the context an IME would see, as in the eval set."""
    n = 0
    for _, text in wiki_articles():
        pos = 0
        for m in LINE_SPLIT.finditer(text + "\n"):
            s = text[pos : m.start()]
            lead = len(s) - len(s.lstrip())
            if len(s.strip()) >= 4:
                start = pos + lead
                yield text[max(0, start - prefix_chars) : start], s.strip()
                n += len(s.strip())
            pos = m.end()
        if n >= max_chars:
            return


def wiki(max_chars=60_000_000):
    n = 0
    for _, text in wiki_articles():
        for s in LINE_SPLIT.split(text):
            s = s.strip()
            if len(s) >= 4:
                yield s
                n += len(s)
        if n >= max_chars:
            return
