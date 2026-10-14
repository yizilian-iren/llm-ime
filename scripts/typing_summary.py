"""Where typing was hard, a day at a time (5A.2, D-030): reads the shell's typing log and writes a Markdown summary.

  python scripts/typing_summary.py [--dir DIR] [DAY ...]

DIR: the log folder (default ~/Library/Application Support/llm-ime/log, where the input method writes one
YYYY-MM-DD.jsonl per day, shell/Sources/TypingLog.swift). Writes YYYY-MM-DD.md beside each day's log; with no
DAY, every day whose summary is missing or older than its log. Runs hourly on M3 (com.llm-ime.summary).
Standard library only. Everything stays on this Mac.
"""

import argparse
import collections
import datetime
import json
import os
import sys

DEFAULT_DIR = os.path.expanduser("~/Library/Application Support/llm-ime/log")
LONG_MS = 2000       # last key to commit: deciding / looking for the right one
WAIT_MS = 300        # a key to its first candidates, or a commit key waiting for the answer (4A.8)
RETYPE_S = 10        # given up (Esc, all deleted) and typed again within this: a retype
EXAMPLES = 8
PAGE = 5

CATEGORIES = [
    ("pick", "没选第 1 个 / 翻了页", "选的不是第 1 个，或翻过页、移过选择"),
    ("retype", "打了又删、重打", "拼音打了又删再打，或 Esc / 删光以后 10 s 内重打"),
    ("deleted_after", "上屏后马上删", "上屏后紧接着按了删除"),
    ("enter", "Enter 上原字母", "没有一个候选对，上了字母"),
    ("long", "用了很久", f"最后一键到上屏超过 {LONG_MS // 1000} s"),
    ("wait", "等候选", f"某个键等第一批候选超过 {WAIT_MS} ms，或上屏键等答案超过 {WAIT_MS} ms"),
]


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short when the process was killed
    return out


def when(r):
    return datetime.datetime.fromisoformat(r["t"].replace("Z", "+00:00"))


def letters_then(events):
    """Whether letters were typed after a delete: keys typed again."""
    seen_del = False
    for e in events:
        if e[1] == "del":
            seen_del = True
        elif seen_del and len(e) == 2 and len(e[1]) == 1:
            return True
    return False


def last_key_ms(r):
    keys = [e[0] for e in r["events"] if len(e) == 2 and (len(e[1]) == 1 or e[1] == "del")]
    return r["ms"] - (keys[-1] if keys else 0)


def first_waits(r):
    """ms from each key to its first candidates (the first answer for that request)."""
    seen, out = set(), []
    for n, stage, ms in r.get("answers", []):
        if n not in seen:
            seen.add(n)
            out.append(ms)
    return out


def classify(records):
    """{category: [record, ...]}; a record can be in several."""
    cats = collections.defaultdict(list)
    for i, r in enumerate(records):
        committed = r.get("text") is not None
        events = r.get("events", [])
        if committed and r["how"] != "enter" and (
                (r.get("picked") or 0) > 0 or any(e[1] in ("page", "move") for e in events)):
            cats["pick"].append(r)
        retyped = letters_then(events)
        if r["how"] in ("esc", "deleted") and i + 1 < len(records):
            nxt = records[i + 1]
            retyped = retyped or (when(nxt) - when(r)).total_seconds() - nxt["ms"] / 1000 <= RETYPE_S
        if retyped:
            cats["retype"].append(r)
        if committed and r.get("deleted_after", 0) > 0:
            cats["deleted_after"].append(r)
        if r["how"] == "enter":
            cats["enter"].append(r)
        if committed and last_key_ms(r) > LONG_MS:
            cats["long"].append(r)
        waits = first_waits(r) + [e[2] for e in events if e[1] == "wait"]
        if waits and max(waits) > WAIT_MS:
            cats["wait"].append(r)
    return cats


def describe(r):
    t = when(r).astimezone().strftime("%H:%M")
    app = (r.get("app") or "?").split(".")[-1]
    ctx = r.get("context") or ""
    ctx = "…" + ctx[-12:] if len(ctx) > 12 else ctx
    bits = []
    p = r.get("picked")
    if p is not None and r.get("text") is not None and r["how"] != "enter":
        bits.append(f"第 {p + 1} 个")
    pages = sum(1 for e in r["events"] if e[1] == "page" and e[2] > 0)
    if pages:
        bits.append(f"翻 {pages} 页")
    dels = sum(1 for e in r["events"] if e[1] == "del")
    if dels:
        bits.append(f"删 {dels} 键")
    if r.get("stage") == "quick" and r.get("text") is not None and r["how"] != "enter":
        bits.append("选时还是快猜")
    if r.get("used"):
        bits.append(f"只上前 {r['used']} 键")
    if r.get("deleted_after"):
        bits.append(f"上屏后删 {r['deleted_after']}")
    waits = first_waits(r) + [e[2] for e in r["events"] if e[1] == "wait"]
    if waits:
        bits.append(f"最久等 {max(waits)} ms")
    bits.append(f"{r['ms'] / 1000:.1f} s，最后一键后 {last_key_ms(r) / 1000:.1f} s")
    typed = "".join("⌫" if e[1] == "del" else e[1] for e in r["events"] if len(e) == 2)
    got = r.get("text") if r.get("text") is not None else f"（{r['how']}）"
    cands = " / ".join(r.get("cands", [])[:PAGE])
    line = f"- {t} {app}  `{r['keys']}` → **{got}**（{'，'.join(bits)}）"
    if typed != r["keys"]:
        line += f"  按键 `{typed}`"
    line += f"\n  前文「{ctx}」 候选：{cands or '无'}"
    return line


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summarize(day, records):
    cats = classify(records)
    commits = [r for r in records if r.get("text") is not None]
    out = [f"# {day} 哪里费力", ""]
    chars = sum(len(r["text"]) for r in commits if r["how"] != "enter")
    given_up = sum(1 for r in records if r["how"] in ("esc", "deleted"))
    out.append(f"上屏 {len(commits)} 次，{chars} 字；放弃（Esc / 删光）{given_up} 次。")
    apps = collections.Counter((r.get("app") or "?") for r in commits)
    if apps:
        out.append("App：" + "，".join(f"{a} {n}" for a, n in apps.most_common(6)))
    quick = [w for r in records for w in first_waits(r)]
    full = [ms for r in records for n, s, ms in r.get("answers", []) if s == "full"]
    if quick:
        out.append(f"每键到第一批候选 p50 {pct(quick, .5)} / p90 {pct(quick, .9)} ms；"
                   f"细算 p50 {pct(full, .5)} / p90 {pct(full, .9)} ms。")
    out += ["", "| 费力点 | 次数 | 占上屏 |", "|---|---|---|"]
    for key, name, _ in CATEGORIES:
        n = len(cats.get(key, []))
        share = f"{100 * n / len(commits):.0f}%" if commits else "-"
        out.append(f"| {name} | {n} | {share} |")
    for key, name, what in CATEGORIES:
        rs = cats.get(key, [])
        if not rs:
            continue
        out += ["", f"## {name}（{len(rs)}）", "", what + "。", ""]
        again = collections.Counter((r["keys"], r.get("text")) for r in rs)
        rep = [(k, t, n) for (k, t), n in again.most_common() if n > 1]
        if rep:
            out.append("反复出现：" + "，".join(f"`{k}` → {t or '（放弃）'} ×{n}" for k, t, n in rep[:10]))
            out.append("")
        out += [describe(r) for r in rs[-EXAMPLES:]]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("days", nargs="*", help="YYYY-MM-DD; default: every day whose summary is out of date")
    a = ap.parse_args()
    if not os.path.isdir(a.dir):
        return
    days = a.days
    if not days:
        for f in sorted(os.listdir(a.dir)):
            if f.endswith(".jsonl"):
                log, md = os.path.join(a.dir, f), os.path.join(a.dir, f[:-6] + ".md")
                if not os.path.exists(md) or os.path.getmtime(md) < os.path.getmtime(log):
                    days.append(f[:-6])
    for day in days:
        log = os.path.join(a.dir, day + ".jsonl")
        if not os.path.exists(log):
            print(f"no log for {day}", file=sys.stderr)
            continue
        md = os.path.join(a.dir, day + ".md")
        with open(md + ".tmp", "w", encoding="utf-8") as f:
            f.write(summarize(day, load(log)))
        os.chmod(md + ".tmp", 0o600)
        os.replace(md + ".tmp", md)
        print(md)


if __name__ == "__main__":
    main()
