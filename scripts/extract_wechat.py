"""Pull the user's own WeChat text messages out of the decrypted Mac WeChat 4.x databases.

Input: data/private/wechat_db/message/message_N.db (decrypted with ~/tools/wechat-db-decrypt-macos, see plan 2A.5).
Only messages the user sent (real_sender_id = the user's row in Name2Id) of type text; others' messages are not read.
Output: data/private/wechat_mine.jsonl, one {"chat", "time", "text"} per message, chat = hash of the chat id.
"""

import glob
import hashlib
import json
import os
import re
import sqlite3

ME = "wxid_YOUR_ID"
HAN = re.compile(r"[一-鿿]")


def decode(content):
    if isinstance(content, bytes):
        try:
            import zstandard
            content = zstandard.ZstdDecompressor().decompress(content, max_output_size=1 << 20)
        except Exception:
            return None
        content = content.decode("utf-8", "ignore")
    return content


def main():
    out, chats = [], {}
    for db in sorted(glob.glob("data/private/wechat_db/message/message_[0-9]*.db")):
        c = sqlite3.connect(db)
        me = c.execute("select rowid from Name2Id where user_name=?", (ME,)).fetchone()
        if not me:
            continue
        # Msg_<md5(chat user_name)>: map back to a stable chat key without keeping the id itself
        for (name,) in c.execute("select user_name from Name2Id"):
            chats["Msg_" + hashlib.md5(name.encode()).hexdigest()] = hashlib.sha1(name.encode()).hexdigest()[:10]
        tables = [r[0] for r in c.execute("select name from sqlite_master where type='table' and name like 'Msg_%'")]
        for t in tables:
            q = f"select create_time, message_content from [{t}] where real_sender_id=? and local_type=1"
            for ts, content in c.execute(q, me):
                text = decode(content)
                if not text or text.startswith("<"):  # xml payloads (quotes, cards) are not typed text
                    continue
                out.append({"chat": chats.get(t, t[4:14]), "time": ts, "text": text.strip()})
    out.sort(key=lambda r: r["time"])
    seen, uniq = set(), []
    for r in out:  # the same message can sit in both message_N.db after an import
        k = (r["chat"], r["time"], r["text"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    with open("data/private/wechat_mine.jsonl", "w") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    han = sum(len(HAN.findall(r["text"])) for r in uniq)
    with_han = sum(1 for r in uniq if HAN.search(r["text"]))
    print(f"{len(uniq)} messages ({with_han} with Chinese), {han} Han chars, {len(set(r['chat'] for r in uniq))} chats")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
