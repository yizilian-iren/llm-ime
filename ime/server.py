"""The engine as a local service for the input method shell (4A.1).

  .venv/bin/python -m ime.server [--sock PATH] [--engine v02]

Loads the engine, then listens on a Unix socket (default ~/Library/Application Support/llm-ime/engine.sock).
One JSON object per line, both ways:
  -> {"id": 7, "context": "下班以后，", "keys": "jintian"}      (op "type" may be left out)
  <- {"id": 7, "stage": "quick", "candidates": [...]}           then the same id with stage "full" (--k of them),
                                                                 and after a pause "more" (--more: full's first),
                                                                 with "segments": [["今天", 6], ...]: candidates for
                                                                 the start of the keys and how many keys each uses
  -> {"op": "commit", "text": "今天"}                            what was committed; kept in memory as context
  -> {"op": "learn", "keys": "liji", "text": "力积", "first": "理解"}   a pick to learn from (5A.3, ime.userwords)
  -> {"op": "forget"}                                           clears the word list (the menu's 删除全部记录)
                                                                 and deletes the personal model (6A.2)
  -> {"op": "hurry"}                                            the full answer to this connection's latest keys is
                                                                 waited for (space): start it now (ime.session)
  -> {"op": "import", "path": "/x/words.tsv", "count": 1, "id": 9}   adds `keys<TAB>word[<TAB>count]` lines
                                                                 <- {"id": 9, "imported": n}
  -> {"op": "ping", "id": 8}   <- {"id": 8, "op": "pong"}
A request whose context is missing or null uses the text committed through this service since it started.
Each connection gets answers to its newest request only: its older ones still being converted are dropped
(ime.session, tagged by connection, so a second client does not starve the shell). All model work runs on the
Session's one worker thread, whichever connection asked.

Nothing the user types is logged; the log has times, connections and errors only. The one thing written to disk is
the user's word list (--words, only on this Mac, D-019).
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
from ime import userwords  # noqa: E402
from ime.session import Session  # noqa: E402

DEFAULT_SOCK = os.path.expanduser("~/Library/Application Support/llm-ime/engine.sock")
CONTEXT_CHARS = 64


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def log_error(where):
    """The traceback without the exception's message, which may quote what the user typed."""
    etype, _, tb = sys.exc_info()
    frames = "".join(traceback.format_tb(tb)).rstrip()
    log(f"error in {where}: {etype.__name__}\n{frames}")


class Conn:
    def __init__(self, sock):
        self.sock = sock
        self.lock = threading.Lock()
        self.latest = None  # (id, context, keys) of its newest type request

    def send(self, obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
        with self.lock:
            try:
                self.sock.sendall(data)
            except OSError:
                pass  # the shell went away; its reader thread cleans up


class Server:
    def __init__(self, engine, k=5, more=None, words=None):
        self.committed = ""  # recent committed text, in memory only
        self.words = words
        self.personal = getattr(engine, "model_path", "") == engines.PERSONAL
        self.session = Session(engine, self._on_result, k=k, more=more, segments=True, words=words,
                               on_error=lambda: log_error("conversion"))

    def _on_result(self, context, keys, cands, stage, tag, segments=None):
        latest = tag.latest
        if latest is None or (latest[1], latest[2]) != (context, keys):
            return  # a newer request of this connection is queued; only its results matter
        msg = {"id": latest[0], "stage": stage, "candidates": cands}
        if segments is not None:
            msg["segments"] = [[t, e] for t, e in segments]
        tag.send(msg)

    def handle(self, conn, msg):
        op = msg.get("op", "type")
        if op == "type":
            context = msg.get("context")
            if context is None:
                context = self.committed
            context, keys = context[-CONTEXT_CHARS:], msg.get("keys", "")
            conn.latest = (msg.get("id"), context, keys)
            self.session.type(context, keys, tag=conn)
        elif op == "hurry":
            self.session.hurry(tag=conn)
        elif op == "commit":
            self.committed = (self.committed + msg.get("text", ""))[-CONTEXT_CHARS:]
        elif op == "learn" and self.words:
            self.words.learn(msg.get("keys", ""), msg.get("text", ""), msg.get("first", ""))
        elif op == "forget":
            if self.words:
                self.words.forget()
            if engines.forget_personal() and self.personal:  # 6A.2: back to the base, which needs a new process
                log("personal model deleted: restarting on the base")
                threading.Timer(0.5, os._exit, (0,)).start()  # launchd starts it again (KeepAlive)
        elif op == "import" and self.words:
            conn.send({"id": msg.get("id"), "imported": self.words.import_file(msg["path"], int(msg.get("count", 1)))})
        elif op == "ping":
            conn.send({"id": msg.get("id"), "op": "pong"})
        else:
            conn.send({"id": msg.get("id"), "error": f"unknown op {op}"})

    def serve_conn(self, sock):
        conn = Conn(sock)
        log("shell connected")
        buf = b""
        try:
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        self.handle(conn, json.loads(line))
                    except Exception:
                        log_error("request")
        except OSError:
            pass
        finally:
            conn.latest = None
            sock.close()
            log("shell disconnected")


def listen(path):
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    if os.path.exists(path):
        os.unlink(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o077)
    try:
        s.bind(path)
    finally:
        os.umask(old)
    s.listen(8)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--engine", default="v02")
    ap.add_argument("--k", type=int, default=5, help="candidates of the full search")
    ap.add_argument("--more", type=int, default=20, help="candidates after a pause, for the shell's further pages")
    ap.add_argument("--words", default=userwords.DEFAULT_PATH, help="the user's word list (5A.3); '' for none")
    # MLX keeps freed buffers for reuse without a limit: 3.7 GB on M3 for 670 MB of models (4A.7). 256 MB: on M4
    # 1.3 GB after typing bench_keys twice, against 3.0 GB without.
    ap.add_argument("--cache-mb", type=int, default=256, help="MLX buffer cache limit (-1: MLX's default)")
    a = ap.parse_args()
    if a.cache_mb >= 0:
        import mlx.core as mx
        mx.set_cache_limit(a.cache_mb << 20)
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model paths are repo-relative

    t = time.time()
    eng = engines.load(a.engine)
    eng.convert("", "nihao", k=1)  # first call compiles; do it before the shell waits on it
    log(f"loaded {a.engine} ({getattr(eng, 'model_path', '?')}) in {time.time() - t:.1f}s")
    words = userwords.UserWords(eng, a.words) if a.words and hasattr(eng, "score") else None
    server = Server(eng, k=a.k, more=a.more, words=words)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # launchd stops it with SIGTERM: remove the socket
    s = listen(a.sock)
    log(f"listening on {a.sock}")
    try:
        while True:
            sock, _ = s.accept()
            threading.Thread(target=server.serve_conn, args=(sock,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
        if os.path.exists(a.sock):
            os.unlink(a.sock)


if __name__ == "__main__":
    main()
