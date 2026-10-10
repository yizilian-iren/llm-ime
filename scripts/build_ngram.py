"""Count words and word bigrams over the public corpora → data/corpus/ngram.pkl.

Text is split into Han-only runs, segmented with jieba; bigrams never cross a run boundary.
"""

import os
import pickle
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import corpus  # noqa: E402

HAN_RUN = re.compile(r"[一-鿿]+")
MIN_WORD, MIN_BIGRAM = 3, 2
BOS = "<s>"


def count_chunk(lines):
    import jieba

    jieba.setLogLevel(60)
    uni, bi = Counter(), Counter()
    for line in lines:
        for run in HAN_RUN.findall(line):
            prev = BOS
            for w in jieba.lcut(run, HMM=False):
                uni[w] += 1
                bi[(prev, w)] += 1
                prev = w
    return uni, bi


def chunks(it, n=20000):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def source():
    yield from corpus.lccc()
    yield from corpus.wiki()


def main():
    t0 = time.time()
    uni, bi = Counter(), Counter()
    with Pool(os.cpu_count() - 1) as pool:
        for i, (u, b) in enumerate(pool.imap_unordered(count_chunk, chunks(source()))):
            uni.update(u)
            bi.update(b)
            if i % 50 == 0:
                print(f"  chunk {i}  words {len(uni)}  bigrams {len(bi)}  {time.time() - t0:.0f}s", flush=True)
            if i % 200 == 199:  # keep memory bounded: drop singletons as we go
                bi = Counter({k: v for k, v in bi.items() if v > 1})
    vocab = {w: c for w, c in uni.items() if c >= MIN_WORD or len(w) == 1}
    bigrams = {k: v for k, v in bi.items() if v >= MIN_BIGRAM and k[1] in vocab and (k[0] == BOS or k[0] in vocab)}
    with open("data/corpus/ngram.pkl", "wb") as f:
        pickle.dump({"unigrams": vocab, "bigrams": bigrams, "bos_count": sum(c for (a, _), c in bi.items() if a == BOS)}, f)
    print(f"vocab {len(vocab)}, bigrams {len(bigrams)}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
