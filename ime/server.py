"""The engine as a local service for the input method shell (4A.1).

  .venv/bin/python -m ime.server [--sock PATH] [--engine v02]

Loads the engine, then listens on a Unix socket (default ~/Library/Application Support/llm-ime/engine.sock).
One JSON object per line, both ways:
  -> {"id": 7, "context": "下班以后，", "keys": "jintian"}      (op "type" may be left out)
  <- {"id": 7, "stage": "quick", "candidates": [...]}           then the same id with stage "full"
  -> {"op": "commit", "text": "今天"}                            what was committed; kept in memory as context
  -> {"op": "ping", "id": 8}   <- {"id": 8, "op": "pong"}
A request whose context is missing or null uses the text committed through this service since it started.
Only the newest request is answered: older ones still being converted are dropped (ime.session).
All model work runs on the Session's one worker thread, whichever connection asked.

Nothing the user types is logged or written to disk; the log has times, connections and errors only.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ime import engine as engines  # noqa: E402
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

    def send(self, obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
        with self.lock:
            try:
                self.sock.sendall(data)
            except OSError:
                pass  # the shell went away; its reader thread cleans up


class Server:
    def __init__(self, engine, k=5):
        self.lock = threading.Lock()
        self.latest = None  # (conn, id, context, keys) of the newest type request
        self.committed = ""  # recent committed text, in memory only
        self.session = Session(engine, self._on_result, k=k)

    def _on_result(self, context, keys, cands, stage):
        with self.lock:
            latest = self.latest
        if latest is None or (latest[2], latest[3]) != (context, keys):
            return  # a newer request is queued; only its results matter
        conn, rid = latest[0], latest[1]
        conn.send({"id": rid, "stage": stage, "candidates": cands})

    def handle(self, conn, msg):
        op = msg.get("op", "type")
        if op == "type":
            context = msg.get("context")
            if context is None:
                context = self.committed
            context, keys = context[-CONTEXT_CHARS:], msg.get("keys", "")
            with self.lock:
                self.latest = (conn, msg.get("id"), context, keys)
            self.session.type(context, keys)
        elif op == "commit":
            self.committed = (self.committed + msg.get("text", ""))[-CONTEXT_CHARS:]
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
            with self.lock:
                if self.latest and self.latest[0] is conn:
                    self.latest = None
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
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model paths are repo-relative

    t = time.time()
    eng = engines.load(a.engine)
    eng.convert("", "nihao", k=1)  # first call compiles; do it before the shell waits on it
    log(f"loaded {a.engine} in {time.time() - t:.1f}s")
    server = Server(eng)
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
