"""The engine's model work on another machine (7E.1, D-049): home4070's GPU, M3's own engine when it is slow or away.

  python -m ime.remote serve --host 100.100.59.15 [--port 8771] [--token-file ~/.config/llm-ime/remote-token]

serve (on home4070, MLX's CUDA backend, scripts/install_remote4070.sh): the same engine (ime.engine.load "v02", the
same models) answers convert / segments / score over TCP, one JSON object per line both ways, on the Tailscale
address only and only after a hello with the token:
  -> {"op": "hello", "token": "..."}            <- {"op": "hello", "model": "r6-4bit"}  (model_name)
  -> {"id": 3, "op": "convert", "context": "", "keys": "nihao", "k": 5, "beam": null, "mix": true,
      "prefix": [[token, key end], ...] | null, "eos": true}
                                                <- {"id": 3, "cands": [...] | null, "paths": {text: [[t, e], ...]}}
  -> {"id": 4, "op": "segments", "context": "", "keys": "..."}   <- {"id": 4, "segs": [[text, keys used], ...]}
  -> {"id": 5, "op": "score", "context": "", "keys": "...", "texts": [...]}   <- {"id": 5, "scores": [...]}
  -> {"op": "cancel", "id": 3}                  the convert is dropped between beam steps (its answer: cands null)
  -> {"op": "forget"}                           deletes the personal models here too (删除全部记录), then exits
                                                (systemd starts it again on the base)
All model work runs on one worker thread, newest request first. Nothing the user types is logged or written here
(D-019, D-049): the log has times, counts and errors only.

RemoteEngine (on M3, ime.server with ~/Library/Application Support/llm-ime/remote.json: {"host", "port", "token"}):
an Engine that asks the remote first and the local engine when the remote is off (the menu's 远程计算, op "remote"
from the shell), not connected, on another model, or slower than `quick_ms` / `full_ms` (then local for `backoff_s`).
Everything else (prefixes, whole_pinyin, ...) is the local engine's. `last_by`: "remote" / "local", for the shell's
typing log.
"""

import argparse
import json
import os
import queue
import shutil
import socket
import sys
import threading
import time
import traceback

DEFAULT_PORT = 8771
DEFAULT_CONFIG = os.path.expanduser("~/Library/Application Support/llm-ime/remote.json")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, file=sys.stderr, flush=True)


def _lines(sock):
    """The JSON objects a socket sends, one per line, until it closes."""
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                yield json.loads(line)


def _send(sock, lock, obj):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
    with lock:
        sock.sendall(data)


# --- the service (home4070) --------------------------------------------------------------------------------------
class Service:
    def __init__(self, engine, token):
        self.engine, self.token = engine, token
        self.model = model_name(engine)
        self.cancelled = set()
        self.jobs = queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        while True:
            conn, lock, msg = self.jobs.get()
            i = msg.get("id")
            t = time.perf_counter()
            try:
                out = self._run(msg)
            except Exception:
                etype, _, tb = sys.exc_info()  # not the message: it may quote what the user typed
                log(f"error in {msg.get('op')}: {etype.__name__}\n{''.join(traceback.format_tb(tb)).rstrip()}")
                out = {"error": etype.__name__}
            self.cancelled.discard(i)
            try:
                _send(conn, lock, {"id": i, **out})
            except OSError:
                pass
            ms = (time.perf_counter() - t) * 1000
            if ms > 500:
                log(f"slow {msg.get('op')}: {ms:.0f} ms")

    def _run(self, msg):
        op, e = msg["op"], self.engine
        if op == "convert":
            if msg.get("id") in self.cancelled:
                return {"cands": None}
            prefix = [tuple(x) for x in msg["prefix"]] if msg.get("prefix") else None
            cands = e.convert(msg["context"], msg["keys"], k=msg.get("k", 5), beam=msg.get("beam"),
                              mix=msg.get("mix", True), prefix=prefix, eos=msg.get("eos", True),
                              stop=lambda: msg.get("id") in self.cancelled)
            paths = {t: [list(x) for x in e.paths[t]] for t in cands or () if t in e.paths}
            return {"cands": cands, "paths": paths}
        if op == "segments":
            return {"segs": [[t, n] for t, n in e.segments(msg["context"], msg["keys"])]}
        if op == "score":
            return {"scores": e.score(msg["context"], msg["keys"], msg["texts"])}
        raise ValueError(op)

    def serve_conn(self, sock):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        lock = threading.Lock()
        ok = False
        try:
            for msg in _lines(sock):
                op = msg.get("op")
                if not ok:
                    if op != "hello" or msg.get("token") != self.token:
                        log("refused a connection without the token")
                        return
                    ok = True
                    _send(sock, lock, {"op": "hello", "model": self.model})
                    log("client connected")
                elif op == "cancel":
                    self.cancelled.add(msg.get("id"))
                elif op == "forget":
                    from ime import engine as engines
                    engines.forget_personal()
                    log("personal model deleted: restarting on the base")
                    threading.Timer(0.5, os._exit, (0,)).start()
                elif op == "ping":
                    _send(sock, lock, {"op": "pong", "id": msg.get("id")})
                else:
                    self.jobs.put((sock, lock, msg))
        except (OSError, ValueError):
            pass
        finally:
            sock.close()
            if ok:
                log("client disconnected")


def serve(a):
    from ime import engine as engines
    import mlx.core as mx

    mx.set_cache_limit(a.cache_mb << 20)
    mx.set_memory_limit(a.memory_mb << 20)  # beside voice-tts and the user's games on a 12 GB card
    token = open(os.path.expanduser(a.token_file)).read().strip()
    t = time.time()
    eng = engines.load(a.engine)
    for keys in ("nihao", "jintiantianqibucuo", "wmxzqcf"):  # the first calls compile kernels (CUDA: seconds each)
        eng.convert("", keys, k=5)
    log(f"loaded {a.engine} ({getattr(eng, 'model_path', '?')}) on {mx.default_device()} in {time.time() - t:.1f}s")
    svc = Service(eng, token)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.host, a.port))
    s.listen(4)
    log(f"listening on {a.host}:{a.port}")
    while True:
        sock, _ = s.accept()
        threading.Thread(target=svc.serve_conn, args=(sock,), daemon=True).start()


# --- the client (M3) ---------------------------------------------------------------------------------------------
class RemoteEngine:
    """An Engine: `local`'s, with its model work asked of the service at host:port first (see the module's doc)."""

    def __init__(self, local, host, port=DEFAULT_PORT, token="", quick_ms=150, full_ms=600, backoff_s=20.0,
                 connect_s=1.0, retry_s=5.0):
        self.local, self.host, self.port, self.token = local, host, int(port), token
        self.quick_ms, self.full_ms, self.backoff_s = quick_ms, full_ms, backoff_s
        self.connect_s, self.retry_s = connect_s, retry_s
        self.on = True  # the menu's 远程计算 (op "remote")
        self.paths = {}
        self.last_by = "local"
        self.counts = {"remote": 0, "local": 0, "timeouts": 0, "errors": 0}
        self._sock = None
        self._lock = threading.Lock()  # sending
        self._cond = threading.Condition()
        self._answers = {}
        self._next = 0
        self._until = 0.0  # local only until then (a timeout)
        self._model = None  # the remote's model, from its hello
        self._wake = threading.Event()
        threading.Thread(target=self._keep_connected, daemon=True).start()

    def __getattr__(self, name):  # what is not model work: the local engine's
        return getattr(self.local, name)

    @property
    def remote_ok(self):
        return self.on and self._sock is not None and time.monotonic() >= self._until

    def _keep_connected(self):
        while True:
            if self._sock is None:
                try:
                    self._connect()
                except OSError:
                    pass
            self._wake.wait(self.retry_s)
            self._wake.clear()

    def _connect(self):
        s = socket.create_connection((self.host, self.port), timeout=self.connect_s)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(None)
        try:
            _send(s, self._lock, {"op": "hello", "token": self.token})
            lines = _lines(s)
            s.settimeout(30)  # its first answer only: it may still be loading
            hello = next(lines)
            s.settimeout(None)
        except (OSError, ValueError, StopIteration):  # refused (the token) or gone
            s.close()
            return
        want = model_name(self.local)
        if hello.get("model") != want:
            s.close()
            if self._model != hello.get("model"):
                log(f"remote runs {hello.get('model')}, here {want}: not used")
            self._model = hello.get("model")
            return
        self._model = hello.get("model")
        self._sock = s
        log(f"remote {self.host}:{self.port} connected ({self._model})")
        threading.Thread(target=self._read, args=(s, lines), daemon=True).start()

    def _read(self, s, lines):
        try:
            for msg in lines:
                with self._cond:
                    self._answers[msg.get("id")] = msg
                    self._cond.notify_all()
        except (OSError, ValueError):
            pass
        self._drop(s)

    def _drop(self, s):
        with self._cond:
            if self._sock is s:
                self._sock = None
                log("remote disconnected")
            self._cond.notify_all()
        try:
            s.close()
        except OSError:
            pass

    def _ask(self, msg, ms, stop=None):
        """The service's answer to msg, None if it took longer than ms (or the connection broke): then the local
        engine answers; "stopped" when stop() said so while waiting."""
        s = self._sock
        if s is None:
            return None
        self._next += 1
        i = msg["id"] = self._next
        try:
            _send(s, self._lock, msg)
        except OSError:
            self._drop(s)
            return None
        end = time.monotonic() + ms / 1000
        with self._cond:
            while i not in self._answers and self._sock is s:
                left = end - time.monotonic()
                if left <= 0:
                    break
                if stop and stop():
                    self._cancel(s, i)
                    return "stopped"
                self._cond.wait(min(left, 0.002) if stop else left)
            ans = self._answers.pop(i, None)
        if ans is None:
            if self._sock is s:
                self._cancel(s, i)
                self.counts["timeouts"] += 1
                self._until = time.monotonic() + self.backoff_s
                log(f"remote slower than {ms} ms: local for {self.backoff_s:.0f} s")
            return None
        if "error" in ans:
            self.counts["errors"] += 1
            return None
        return ans

    def _cancel(self, s, i):
        try:
            _send(s, self._lock, {"op": "cancel", "id": i})
        except OSError:
            pass
        threading.Timer(5, lambda: self._answers.pop(i, None)).start()  # its late answer, if any

    def convert(self, context, keys, k=5, beam=None, mix=True, prefix=None, stop=None, eos=True):
        if self.remote_ok and keys:
            quick = (beam or 99) <= 3
            ans = self._ask({"op": "convert", "context": context, "keys": keys, "k": k, "beam": beam, "mix": mix,
                             "prefix": [list(x) for x in prefix] if prefix else None, "eos": eos},
                            self.quick_ms if quick else self.full_ms, stop)
            if ans == "stopped":
                return None
            if ans is not None and (ans["cands"] is not None or (stop and stop())):
                self.paths = {t: tuple(tuple(x) for x in p) for t, p in ans.get("paths", {}).items()}
                self.last_by = "remote"
                self.counts["remote"] += 1
                return ans["cands"]
        out = self.local.convert(context, keys, k=k, beam=beam, mix=mix, prefix=prefix, stop=stop, eos=eos)
        self.paths = self.local.paths
        self.last_by = "local"
        self.counts["local"] += 1
        return out

    def segments(self, context, keys, **kw):
        if self.remote_ok and not kw:
            ans = self._ask({"op": "segments", "context": context, "keys": keys}, self.full_ms)
            if ans is not None:
                return [(t, n) for t, n in ans["segs"]]
        return self.local.segments(context, keys, **kw)

    def score(self, context, keys, texts):
        if self.remote_ok and texts:
            ans = self._ask({"op": "score", "context": context, "keys": keys, "texts": list(texts)}, self.full_ms)
            if ans is not None:
                return ans["scores"]
        return self.local.score(context, keys, texts)

    def set_on(self, on):
        self.on = bool(on)
        if self.on:
            self._until = 0.0
            self._wake.set()

    def forget(self):
        s = self._sock
        if s is not None:
            try:
                _send(s, self._lock, {"op": "forget"})
            except OSError:
                pass


def model_name(engine):
    """The folder name of the engine's model, through the personal-4bit link (personal-20261019-4bit): both sides
    must run the very same weights."""
    return os.path.basename(os.path.realpath(getattr(engine, "model_path", "")))


def load_config(path=DEFAULT_CONFIG):
    """remote.json's settings, or None when there is none."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--host", required=True, help="the Tailscale address to listen on (never 0.0.0.0)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--token-file", default="~/.config/llm-ime/remote-token")
    p.add_argument("--engine", default="v02")
    p.add_argument("--cache-mb", type=int, default=256, help="MLX buffer cache limit")
    p.add_argument("--memory-mb", type=int, default=1536, help="MLX memory limit (two 4-bit 0.6B models: ~0.7 GB)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # model paths are repo-relative
    serve(a)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
