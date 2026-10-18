"""Imports the words the user picked in Rime (rime_ice.userdb) into the engine service's word list (5A.7, D-037).

  .venv/bin/python scripts/rime_import.py [--db ~/Library/Rime/rime_ice.userdb]            counts only
  .venv/bin/python scripts/rime_import.py --apply [--quiet-min 10]                         imports them
  .venv/bin/python scripts/rime_import.py --check 30                                       ranks of imported words

Runs on the Mac with the Rime data and the service (M3), and prints counts and ranks only, never a word (D-019).
Reads a copy of the LevelDB (pure Python, snappy included: Rime need not be installed) and keeps the rule of 5A.6:
picked at least twice, 2-6 characters the engine has readings for, the syllables read them, not in jieba's
dictionary (the model mostly types those right already, and 1.9k bonuses of 10-16 nats would override the context),
not in the word list yet. Keys are Rime's syllables joined, count min(c, 3), last used the import (Rime's t is a
commit counter, not a time).

--apply refuses while the typing log was written in the last --quiet-min minutes, copies words.json to
words.json.bak-rime-<time> beside it, and has the service import the lines (op import), so that the service's copy
of the list, which it saves on every pick, is the one that has them. Run again, it finds them in the list and adds
nothing. --check types the keys of some imported words into the service with no context and counts where the word
is in the quick and the full answer.
"""

import argparse
import collections
import glob
import json
import os
import random
import shutil
import socket
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime.ngram import char_readings_table  # noqa: E402
from ime.userwords import DEFAULT_PATH  # noqa: E402

DEFAULT_DB = os.path.expanduser("~/Library/Rime/rime_ice.userdb")
LOG_DIR = os.path.join(os.path.dirname(DEFAULT_PATH), "log")
DEFAULT_SOCK = os.path.join(os.path.dirname(DEFAULT_PATH), "engine.sock")  # as ime.server, which loads the model
MAX_COUNT = 3


# --- LevelDB, read only ---------------------------------------------------------------------------------------------

def varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        s += 7
        if x < 0x80:
            return r, i


def unsnappy(b):
    n, i = varint(b, 0)
    out = bytearray()
    while i < len(b):
        t = b[i]
        i += 1
        kind = t & 3
        if kind == 0:  # literal
            n_lit = t >> 2
            if n_lit >= 60:
                nb = n_lit - 59
                n_lit = int.from_bytes(b[i : i + nb], "little")
                i += nb
            n_lit += 1
            out += b[i : i + n_lit]
            i += n_lit
            continue
        if kind == 1:
            length, off = ((t >> 2) & 7) + 4, ((t >> 5) << 8) | b[i]
            i += 1
        elif kind == 2:
            length, off = (t >> 2) + 1, int.from_bytes(b[i : i + 2], "little")
            i += 2
        else:
            length, off = (t >> 2) + 1, int.from_bytes(b[i : i + 4], "little")
            i += 4
        for _ in range(length):  # copies may overlap what they write
            out.append(out[-off])
    assert len(out) == n
    return bytes(out)


def block(f, off, size):
    raw = f[off : off + size]
    return unsnappy(raw) if f[off + size] == 1 else raw


def block_entries(b):
    restarts = struct.unpack("<I", b[-4:])[0]
    end, i, key = len(b) - 4 - 4 * restarts, 0, b""
    while i < end:
        shared, i = varint(b, i)
        unshared, i = varint(b, i)
        vlen, i = varint(b, i)
        key = key[:shared] + b[i : i + unshared]
        i += unshared
        yield key, b[i : i + vlen]
        i += vlen


def table(path):
    """(user key, sequence, type: 1 value / 0 deletion, value) of one .ldb file."""
    with open(path, "rb") as fh:
        f = fh.read()
    footer = f[-48:]
    assert footer[-8:] == struct.pack("<Q", 0xDB4775248B80FB57), path
    _, j = varint(footer, 0)
    _, j = varint(footer, j)
    index_off, j = varint(footer, j)
    index_size, _ = varint(footer, j)
    for _, handle in block_entries(block(f, index_off, index_size)):
        off, k = varint(handle, 0)
        size, _ = varint(handle, k)
        for ikey, v in block_entries(block(f, off, size)):
            tag = int.from_bytes(ikey[-8:], "little")
            yield ikey[:-8], tag >> 8, tag & 0xFF, v


def rime_rows(db):
    """Rime's user words [(syllables, word, commits)] from a copy of the LevelDB at `db` (the copy then deleted)."""
    with tempfile.TemporaryDirectory() as d:
        copy = os.path.join(d, "userdb")
        shutil.copytree(db, copy)
        logs = glob.glob(os.path.join(copy, "*.log"))
        if any(os.path.getsize(p) for p in logs):
            sys.exit("the userdb has writes not yet in a table (.log not empty): open and quit Rime once, or add a log reader")
        latest = {}
        for p in sorted(glob.glob(os.path.join(copy, "*.ldb"))):
            for k, seq, typ, v in table(p):
                if k not in latest or seq > latest[k][0]:
                    latest[k] = (seq, typ, v)
    rows = []
    for k, (_seq, typ, v) in latest.items():
        ks = k.decode("utf-8", "replace")
        if typ != 1 or ks.startswith("\x01/"):  # deleted, or Rime's own metadata
            continue
        code, _, word = ks.partition("\t")
        attrs = dict(x.split("=", 1) for x in v.decode("utf-8", "replace").split() if "=" in x)
        rows.append((code.split(), word, int(attrs.get("c", 0))))
    return rows


# --- the rule -------------------------------------------------------------------------------------------------------

def norm(s):
    s = s.replace("ü", "v")
    return "lve" + s[3:] if s.startswith("lue") else "nve" + s[3:] if s.startswith("nue") else s


def jieba_words():
    import jieba
    out = set()
    with open(os.path.join(os.path.dirname(jieba.__file__), "dict.txt"), encoding="utf-8") as f:
        for line in f:
            out.add(line.split(" ", 1)[0])
    return out


def load_words(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select(rows, words, readings, jieba_dict):
    """The rows to import [(keys, word, count)], and how many each step of the rule kept."""
    have = {(k, w) for k, ws in words.items() for w in ws}
    steps = collections.Counter()
    out, seen = [], set()
    for syls, word, c in rows:
        steps["userdb"] += 1
        if c < 2:
            continue
        steps["c>=2"] += 1
        if not 2 <= len(word) <= 6:
            continue
        steps["2-6 characters"] += 1
        if not all(ch in readings for ch in word):
            continue
        steps["engine has the characters"] += 1
        if len(syls) != len(word) or any(norm(s) not in readings[ch] for s, ch in zip(syls, word)):
            continue
        steps["syllables read them"] += 1
        if word in jieba_dict:
            continue
        steps["not in jieba"] += 1
        keys = "".join(norm(s) for s in syls)
        if not (keys.isascii() and keys.isalpha() and keys.islower()) or (keys, word) in seen:
            continue
        seen.add((keys, word))
        if (keys, word) in have:
            continue
        steps["not in words.json"] += 1
        out.append((keys, word, min(c, MAX_COUNT)))
    return out, steps


# --- the service ----------------------------------------------------------------------------------------------------

def connect(sock):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(60)
    s.connect(sock)
    return s, s.makefile("rb")


def send(s, obj):
    s.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode())


def log_quiet_for():
    """Seconds since the typing log was last written (inf with no log)."""
    times = [os.path.getmtime(p) for p in glob.glob(os.path.join(LOG_DIR, "*"))]
    return time.time() - max(times) if times else float("inf")


def apply(todo, a):
    quiet = log_quiet_for()
    if quiet < a.quiet_min * 60:
        sys.exit(f"the typing log was written {quiet / 60:.1f} min ago (< {a.quiet_min}): not now")
    before = load_words(a.words)
    backup = f"{a.words}.bak-rime-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(a.words, backup)
    os.chmod(backup, 0o600)
    print(f"backup: {backup}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rime.tsv")
        with open(path, "w", encoding="utf-8") as f:
            for keys, word, c in todo:
                f.write(f"{keys}\t{word}\t{c}\n")
        os.chmod(path, 0o600)
        s, f = connect(a.sock)
        send(s, {"op": "import", "id": 1, "path": path, "count": 1})
        for line in f:
            r = json.loads(line)
            if r.get("id") == 1:
                break
        s.close()
    after = load_words(a.words)
    pairs = lambda w: sum(len(ws) for ws in w.values())  # noqa: E731
    print(f"service imported {r.get('imported', r)}; words.json pairs {pairs(before)} -> {pairs(after)} "
          f"(+{pairs(after) - pairs(before)}), keys {len(before)} -> {len(after)}")


def rank(cands, w):
    return cands.index(w) + 1 if w in cands else None


def check(rows, a, readings, jieba_dict):
    """Ranks of n imported words (in words.json now) in the service's quick and full answers, no context."""
    words = load_words(a.words)
    have = {(k, w) for k, ws in words.items() for w in ws}
    every, _ = select(rows, {}, readings, jieba_dict)
    mine = [(k, w) for k, w, _c in every if (k, w) in have]
    sample = random.Random(a.seed).sample(mine, min(a.check, len(mine)))
    s, f = connect(a.sock)
    ranks = {"quick": collections.Counter(), "full": collections.Counter()}
    ms = []
    for i, (keys, word) in enumerate(sample, 1):
        t = time.perf_counter()
        send(s, {"id": i, "context": "", "keys": keys})
        got = set()
        while got != {"quick", "full"}:
            r = json.loads(f.readline())
            if r.get("id") == i and r.get("stage") in ranks and r["stage"] not in got:
                got.add(r["stage"])
                n = rank(r["candidates"], word)
                ranks[r["stage"]]["1" if n == 1 else "2-5" if n and n <= 5 else "later" if n else "missing"] += 1
        ms.append((time.perf_counter() - t) * 1000)
    s.close()
    print(f"imported words in words.json now: {len(mine)}; checked {len(sample)} (seed {a.seed}), no context")
    for stage, c in ranks.items():
        print(f"  {stage}: " + ", ".join(f"{k} {c[k]}" for k in ("1", "2-5", "later", "missing")))
    if ms:
        ms.sort()
        print(f"  full answer after {ms[len(ms) // 2]:.0f} ms (median)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--words", default=DEFAULT_PATH)
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--quiet-min", type=float, default=10)
    ap.add_argument("--check", type=int, default=0, help="how many imported words to type into the service")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = rime_rows(a.db)
    readings = {ch: {norm(r) for r in rs} for ch, rs in char_readings_table().items()}
    jieba_dict = jieba_words()
    if a.check:
        check(rows, a, readings, jieba_dict)
        return
    todo, steps = select(rows, load_words(a.words), readings, jieba_dict)
    print("kept: " + ", ".join(f"{k} {v}" for k, v in steps.items()))
    print("to import:", len(todo), "count", dict(sorted(collections.Counter(c for _k, _w, c in todo).items())),
          "characters", dict(sorted(collections.Counter(len(w) for _k, w, _c in todo).items())),
          "keys with two or more", sum(v > 1 for v in collections.Counter(k for k, _w, _c in todo).values()))
    if a.apply and todo:
        apply(todo, a)


if __name__ == "__main__":
    main()
