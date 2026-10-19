"""The weekly retrain (6A.2, N-009 layer 3, D-038): a LoRA on the user's commits, deployed to M3 if it does better.

  ~/Library/Application\\ Support/llm-ime-retrain/code/.venv/bin/python scripts/retrain.py [--now]

Runs on M4, every hour (launchd, scripts/install_retrain.sh); each run does what the week's run is waiting for and
exits. A week's run starts Sunday 03:30 and goes through these stages, each retried the next hour while it waits:
  fetch    M3 (a laptop: may be closed) prints the samples (scripts/retrain_samples.py); not after Monday 09:00.
  train    at night (00:00-09:00) when home4070's GPU is idle (the user games on it): samples -> npz there
           (scripts/retrain_npz.py), a LoRA on the deployed base (ime.engine.base_model) with as many public
           samples replayed, the merged model back to M4 as MLX 4-bit. Not after Monday 09:00.
  eval     on M4: the first candidate on the test days (scripts/retrain_eval.py) and run_eval.py --limit 100, the
           new model against the base.
  deploy   if better: when M3 is up (a restart is 3 s, so not waiting for the user to stop typing), the model goes
           to M3's models/personal-<day>-4bit, models/personal-4bit points at it (ime.engine.personal_or), the
           engine service restarts. Until the next week's run starts.
The test days: the newest third of the days in the log while it has fewer than 14, then the newest 7. Training
data lives only on M3, M4 and home4070 (D-038) and is deleted after; the adapter and the numbers stay in
runs/<day>/. Every outcome is a section of M3's log/retrain.md (counts only, beside the daily summaries, where the
main agent reads them); a failure also shows a notification on M3. The input method's menu item 暂停每周重训
(retrainPaused) skips the week. --now: start a new run now (its training still waits for the night and an
idle GPU). --now --base r6-4bit --hold: train on another base and keep the result in ROOT/held/
personal-<day>-4bit instead of deploying (a new base whose engine switch is not merged yet, D-046).
"""

import argparse
import datetime
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = os.path.expanduser("~/Library/Application Support/llm-ime-retrain")
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ROOT/code, or a checkout
STATE = os.path.join(ROOT, "state.json")
SSH = ["ssh", "-o", "ClearAllForwardings=yes", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"]
M3, GPU = "m3", "home4070"
M3_BASE = "$HOME/Library/Application Support/llm-ime"  # as scripts/deploy_m3.sh
M3_MODELS = "$HOME/Projects/llm-ime/models"
M3_PY = "$HOME/Projects/llm-ime/.venv/bin/python"
GPU_DIR = "llm-ime"  # home4070: .venv-train, models/<base>-hf, data/replay/<base>.npz
LABEL = "com.llm-ime.engine"
BUNDLE = "com.iren.inputmethod.LLMIME"
TRAIN_ARGS = ["--lora", "16", "--ckpt", "--tokens", "4096", "--ce-chunk", "256", "--mem-frac", "0.30",  # 6A.1
              "--lr", "1e-4", "--warmup", "20", "--epochs", "3", "--eval-every", "100000", "--replay-ratio", "1"]
GPU_IDLE = (15, 8500)  # utilization % and MiB used at most (voice-tts and Windows hold ~7.45 GB)
NIGHT = (0, 9)  # hours training may start in
MIN_TRAIN = 500  # samples


class Wait(Exception):
    """Not now: try again next hour."""


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def ssh(host, cmd, **kw):
    return sh(SSH + [host, cmd], **kw)


def up(host):
    try:
        ssh(host, "true", timeout=40)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"notes": []}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def week_start(now):
    """The Sunday 03:30 the week's run started at (or starts at)."""
    d = now - datetime.timedelta(days=(now.weekday() + 1) % 7)
    s = d.replace(hour=3, minute=30, second=0, microsecond=0)
    return s if s <= now else s - datetime.timedelta(days=7)


def note(st, text):
    """A line for M3's retrain.md, written the next time M3 is up."""
    st["notes"].append(text)
    log(text)


def flush_notes(st, title=None):
    if not st["notes"] or not up(M3):
        return
    body = (f"\n## {title or datetime.date.today().isoformat()}\n\n" + "".join(f"- {n}\n" for n in st["notes"]))
    ssh(M3, f'mkdir -p "{M3_BASE}/log" && cat >> "{M3_BASE}/log/retrain.md"', input=body)
    st["notes"] = []


def notify(text):
    try:
        ssh(M3, "osascript -e " + shlex.quote(f'display notification "{text}" with title "LLM 输入法：每周重训"'),
            timeout=40)
    except Exception:
        pass


def paused():
    try:
        return ssh(M3, f"defaults read {BUNDLE} retrainPaused 2>/dev/null || true", timeout=40).strip() == "1"
    except subprocess.CalledProcessError:
        return False


# --- the stages ----------------------------------------------------------------------------------------------------

def fetch(run, work):
    if paused():
        raise Skip("paused (暂停每周重训)")
    base = ssh(M3, f'cd "{M3_BASE}/engine" && PYTHONPATH=. {M3_PY} -c '
                   '"from ime.engine import base_model; print(base_model())"').strip()
    run["base"] = run.get("base_override") or os.path.basename(base)  # r2-4bit
    samples = os.path.join(work, "samples.jsonl")
    with open(os.path.join(CODE, "scripts", "retrain_samples.py")) as src, open(samples, "w") as out:
        subprocess.run(SSH + [M3, f'cd "{M3_BASE}/engine" && PYTHONPATH=. {M3_PY} -'], stdin=src, stdout=out,
                       check=True, timeout=600)
    os.chmod(samples, 0o600)
    days, counts = {}, None
    for line in open(samples, encoding="utf-8"):
        s = json.loads(line)
        if "counts" in s:
            counts = s["counts"]
        else:
            days[s["day"]] = days.get(s["day"], 0) + 1
    order = sorted(days)
    if len(order) < 14:
        test = order[len(order) - max(1, round(len(order) / 3)) :]
    else:
        last = datetime.date.fromisoformat(order[-1])
        test = [d for d in order if datetime.date.fromisoformat(d) > last - datetime.timedelta(days=7)]
    run.update(days=len(order), days_list=order, test_days=test, train_n=sum(days[d] for d in order if d not in test),
               test_n=sum(days[d] for d in test), log_counts=counts)
    log(f"fetched: base {run['base']}, {len(order)} days, train {run['train_n']}, test {run['test_n']} ({test})")
    if run["train_n"] < MIN_TRAIN:
        raise Skip(f"too few samples to train on: {run['train_n']} (< {MIN_TRAIN})")


def gpu_idle():
    out = ssh(GPU, "/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")
    util, used = (int(x) for x in out.split(","))
    return util <= GPU_IDLE[0] and used <= GPU_IDLE[1], f"GPU {util}%, {used} MiB"


def train(run, work, now):
    if not NIGHT[0] <= now.hour < NIGHT[1]:
        raise Wait("training waits for the night")
    if not up(GPU):
        raise Wait("home4070 is down")
    idle, what = gpu_idle()
    if not idle:
        raise Wait(f"home4070 is busy: {what}")
    base, hf = run["base"], run["base"].replace("-4bit", "-hf")
    models = os.path.join(CODE, "models")
    if not os.path.isdir(os.path.join(models, hf)):
        raise Skip(f"no models/{hf} on M4 to train on")
    order_file = os.path.join(models, base, "ime_format")
    order = (open(order_file).read().strip() or "ctx_first") if os.path.exists(order_file) else "ctx_first"
    replay = f"data/replay/{base}.npz"
    if ssh(GPU, f"cd {GPU_DIR} && [ -f {replay} ] && echo yes || echo no").strip() != "yes":
        raise Skip(f"no {replay} on home4070 (public samples in {base}'s format)")
    t = time.time()
    rw = f"{GPU_DIR}/retrain-work"
    ssh(GPU, f"rm -rf {rw} && mkdir -m 700 -p {rw}/code {rw}/data")
    sh(["rsync", "-a", "--delete", f"{os.path.join(models, hf)}/", f"{GPU}:{GPU_DIR}/models/{hf}/"])
    archive = subprocess.Popen(["tar", "-C", CODE, "-cf", "-", "ime", "scripts"], stdout=subprocess.PIPE)
    subprocess.run(SSH + [GPU, f"tar -x -C {rw}/code"], stdin=archive.stdout, check=True)
    archive.wait()
    sh(["scp", "-q", "-o", "ClearAllForwardings=yes", os.path.join(work, "samples.jsonl"), f"{GPU}:{rw}/samples.jsonl"])
    npz = ssh(GPU, f"cd {GPU_DIR} && .venv-train/bin/python {rw.split('/', 1)[1]}/code/scripts/retrain_npz.py "
                   f"retrain-work/samples.jsonl --test-days {','.join(run['test_days'])} --tokenizer models/{hf} "
                   f"--order {order} --out retrain-work/data")
    run["npz"] = json.loads(npz.strip().splitlines()[-1])
    log(f"npz: {run['npz']}")
    args = " ".join(["--model", f"models/{hf}", "--data", "retrain-work/data", "--replay", replay,
                     "--out", "retrain-work/run"] + TRAIN_ARGS)
    ssh(GPU, f"cd {GPU_DIR} || exit 1; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup .venv-train/bin/python "
             f"retrain-work/code/scripts/train.py {args} > retrain-work/train.log 2>&1 < /dev/null & echo $! > retrain-work/pid")
    while True:  # ssh may drop; the training goes on
        time.sleep(30)
        try:
            alive = ssh(GPU, f"kill -0 $(cat {rw}/pid) 2>/dev/null && echo yes || echo no").strip() == "yes"
        except subprocess.CalledProcessError:
            continue
        if not alive:
            break
    tail = ssh(GPU, f"tail -c 3000 {rw}/train.log")
    if "\ndone" not in tail:
        raise RuntimeError("training failed: " + tail.strip().splitlines()[-1][:300])
    valid = [line for line in tail.splitlines() if " valid " in line]
    run["valid"] = valid[-1] if valid else ""
    peak = [json.loads(x.replace("'", '"'))["mem_gb"] for x in tail.splitlines() if x.startswith("{'step'")]
    run["train_mem_gb"] = max(peak) if peak else None
    for name in ("final", "final-merged"):
        sh(["rsync", "-a", f"{GPU}:{rw}/run/{name}/", os.path.join(work, name) + "/"])
    for f in ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
        shutil.copy(os.path.join(models, hf, f), os.path.join(work, "final-merged", f))
    new = os.path.join(work, "new-4bit")
    shutil.rmtree(new, ignore_errors=True)
    sh([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", os.path.join(work, "final-merged"),
        "--mlx-path", new, "-q", "--q-bits", "4", "--q-group-size", "64"])
    for f in ("ime_format", "han_tokens.pkl"):
        if os.path.exists(os.path.join(models, base, f)):
            shutil.copy(os.path.join(models, base, f), os.path.join(new, f))
    with open(os.path.join(new, "base"), "w") as f:
        f.write(base + "\n")
    shutil.rmtree(os.path.join(work, "final-merged"))
    ssh(GPU, f"rm -rf {rw}")
    run["train_s"] = round(time.time() - t)
    log(f"trained in {run['train_s']} s: {run['valid']}")


def run_eval(model, tag):
    out = sh([sys.executable, os.path.join(CODE, "scripts", "run_eval.py"), "--engine", "v02", "--opt",
              f"model={model}", "--limit", "100", "--latency", "0", "--tag", tag], cwd=CODE)
    p1 = {}
    for line in out.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in ("full", "initials", "mixed"):
            p1[cells[0]] = float(cells[2])
    return p1


def evaluate(run, work, st):
    t = time.time()
    base, new = f"models/{run['base']}", os.path.join(work, "new-4bit")
    out = sh([sys.executable, os.path.join(CODE, "scripts", "retrain_eval.py"), os.path.join(work, "samples.jsonl"),
              "--days", ",".join(run["test_days"]), "--model", base, "--model", new], cwd=CODE)
    r = json.loads(out.strip().splitlines()[-1])
    run["test"] = {"n": r["n"], "base": r[base], "new": r[new]}
    cache = st.setdefault("run_eval", {})
    if run["base"] not in cache:  # the base's numbers stay the same week to week
        cache[run["base"]] = run_eval(base, f"retrain-{run['base']}")
    run["run_eval"] = {"base": cache[run["base"]], "new": run_eval(new, "retrain-new")}
    worse = [f for f, v in run["run_eval"]["base"].items() if run["run_eval"]["new"].get(f, 0) < v - 3]
    better = run["test"]["new"]["p1"] > run["test"]["base"]["p1"]
    run["better"] = better and not worse
    run["why"] = ("" if better else "not better on the test days") + (f"; run_eval worse on {worse}" if worse else "")
    run["eval_s"] = round(time.time() - t)
    log(f"eval: {run['test']}, {run['run_eval']}, better {run['better']} {run['why']}")


def deploy(run, work):
    if not up(M3):
        raise Wait("M3 is down")
    if paused():
        raise Skip("paused (暂停每周重训) before deploying")
    first = min(d for d in run["days_list"] if d not in run["test_days"])
    if ssh(M3, f'[ -f "{M3_BASE}/log/{first}.jsonl" ] && echo yes || echo no').strip() != "yes":
        raise Skip("the records it was trained on were deleted (删除全部记录): not deployed")
    name = f"personal-{run['day'].replace('-', '')}-4bit"
    sh(["rsync", "-a", "--delete", os.path.join(work, "new-4bit") + "/", f"{M3}:Projects/llm-ime/models/{name}/"])
    engine_log = "$HOME/Library/Logs/llm-ime/engine.log"
    swap = f'cd "{M3_MODELS}" && prev=$(readlink personal-4bit || true) && ln -sfh {name} personal-4bit && echo "$prev"'
    prev = ssh(M3, swap).strip()
    ssh(M3, f"launchctl kickstart -k gui/$(id -u)/{LABEL}")
    loaded = ""
    for _ in range(40):
        time.sleep(3)
        loaded = ssh(M3, f'grep "loaded v02" "{engine_log}" | tail -1').strip()
        if "personal-4bit" in loaded and loaded[:19] >= run["deploy_at"]:
            break
    else:
        back = f"ln -sfh {prev} personal-4bit" if prev else "rm -f personal-4bit"
        ssh(M3, f'cd "{M3_MODELS}" && {back} && launchctl kickstart -k gui/$(id -u)/{LABEL}')
        raise RuntimeError(f"the engine did not load {name} (rolled back to {prev or 'the base'}): {loaded[-120:]}")
    keep = {name, prev}
    ssh(M3, f'cd "{M3_MODELS}" && for d in personal-2*-4bit; do case " {" ".join(keep)} " in *" $d "*) ;; '
            '*) rm -rf "$d";; esac; done')
    run["deployed"] = name
    log(f"deployed {name} (before: {prev or 'the base'})")


def hold(run, work):
    """--base --hold: a personal model on a base M3 does not run yet; it waits in ROOT/held/ for the engine switch."""
    name = f"personal-{run['day'].replace('-', '')}-4bit"
    dst = os.path.join(ROOT, "held", name)
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(os.path.join(work, "new-4bit"), dst)
    run["held"] = dst
    log(f"held {dst} (not deployed; base {run['base']})")


class Skip(Exception):
    """This week's run ends here, with the reason in retrain.md."""


def summary(run):
    t, e = run.get("test"), run.get("run_eval")
    lines = [f"底座 {run.get('base')};样本：训练 {run.get('train_n')}、测试 {run.get('test_n')}"
             f"(测试的天：{', '.join(run.get('test_days', []))})"]
    if t:
        lines.append(f"测试集首位：底座 {t['base']['p1']}% → 新 {t['new']['p1']}%(前 5:{t['base']['p5']}% → "
                     f"{t['new']['p5']}%,{t['n']} 条)")
    if e:
        lines.append("run_eval --limit 100 P@1:" + ", ".join(f"{f} {e['base'][f]} → {e['new'].get(f)}" for f in e["base"]))
    if run.get("train_s") is not None:
        lines.append(f"训练 {run['train_s']} s,峰值 {run.get('train_mem_gb')} GB;{run.get('valid', '')}")
    if run.get("deployed"):
        lines.append(f"已部署 {run['deployed']}")
    elif run.get("held"):
        lines.append(f"没部署，留着等底座切换：{run['held']}")
    elif t:
        lines.append(f"没部署：{run.get('why')}")
    return lines


def step(st, now, force, base=None, hold_it=False):
    run = st.get("run")
    start = week_start(now)
    if force or not run or run["week"] < start.isoformat():
        if run and not run.get("closed"):
            note(st, f"{run['day']} 那周的重训没做完，停在 {run['stage']}:{run.get('waiting', '')}")
        run = st["run"] = {"week": start.isoformat(), "day": now.date().isoformat(), "stage": "fetch"}
        if base:
            run.update(base_override=base, hold=hold_it)
    if run.get("closed"):
        return
    work = os.path.join(ROOT, "work")
    os.makedirs(work, mode=0o700, exist_ok=True)
    deadline = (start + datetime.timedelta(days=1, hours=5, minutes=30)).isoformat()  # Monday 09:00
    if run.get("base_override"):  # --now --base: a run of its own, not the week's
        run.setdefault("started", now.isoformat())
        deadline = (datetime.datetime.fromisoformat(run["started"]) + datetime.timedelta(hours=36)).isoformat()
    try:
        if run["stage"] == "fetch":
            if not up(M3):
                raise Wait("M3 is down")
            fetch(run, work)
            run["stage"] = "train"
            save_state(st)
        if run["stage"] == "train":
            train(run, work, now)
            run["stage"] = "eval"
            save_state(st)
        if run["stage"] == "eval":
            evaluate(run, work, st)
            run["stage"] = "hold" if run.get("hold") else "deploy" if run["better"] else "done"
            save_state(st)
        if run["stage"] == "hold":
            hold(run, work)
            run["stage"] = "done"
            save_state(st)
        if run["stage"] == "deploy":
            run["deploy_at"] = time.strftime("%Y-%m-%d %H:%M:%S")  # as the engine's log
            deploy(run, work)
            run["stage"] = "done"
        if run["stage"] == "done":
            for line in summary(run):
                note(st, line)
            close(st, run, work)
    except (Wait, subprocess.CalledProcessError, subprocess.TimeoutExpired) as w:
        if isinstance(w, subprocess.CalledProcessError) and w.returncode != 255:  # 255: ssh could not connect
            return failed(st, run, work, w)
        run["waiting"] = str(w) if isinstance(w, Wait) else "a machine went away"
        log(f"waiting ({run['stage']}): {w}")
        if run["stage"] in ("fetch", "train") and now.isoformat() > deadline:
            note(st, f"{run['day']} 那周没训：到周一 09:00 还在等({run['stage']}:{w})")
            close(st, run, work)
    except Skip as s:
        note(st, f"{run['day']} 那周没训：{s}")
        close(st, run, work)
    except Exception as e:
        failed(st, run, work, e)


def failed(st, run, work, e):
    """A bug or a broken machine: the week's run ends (once a week is enough to hear of it)."""
    where = traceback.extract_tb(e.__traceback__)[-1]
    detail = (e.stderr or "")[-300:] if isinstance(e, subprocess.CalledProcessError) else str(e)[:300]
    note(st, f"{run['day']} 那周的重训失败，在 {run['stage']}:{type(e).__name__}: {detail.strip()} "
             f"({os.path.basename(where.filename)}:{where.lineno})")
    for line in summary(run):
        note(st, line)
    close(st, run, work)
    notify(f"失败，在 {run['stage']}:详情在 retrain.md")


def close(st, run, work):
    """The run's numbers and adapter stay in runs/<day>/; the samples and models made for it go."""
    keep = os.path.join(ROOT, "runs", run["day"])
    os.makedirs(keep, exist_ok=True)
    if os.path.isdir(os.path.join(work, "final")):
        shutil.copytree(os.path.join(work, "final"), os.path.join(keep, "adapter"), dirs_exist_ok=True)
    with open(os.path.join(keep, "result.json"), "w") as f:
        json.dump(run, f, ensure_ascii=False, indent=1)
    shutil.rmtree(work, ignore_errors=True)
    try:
        ssh(GPU, f"rm -rf {GPU_DIR}/retrain-work", timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        note(st, f"home4070 连不上，{GPU_DIR}/retrain-work(训练数据)没删掉：下次开训时会先删")
    run["closed"] = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true", help="start this week's run now")
    ap.add_argument("--base", help="with --now: train on this base (models/<base>, e.g. r6-4bit), not M3's deployed one")
    ap.add_argument("--hold", action="store_true", help="with --base: keep the result in ROOT/held/, do not deploy")
    a = ap.parse_args()
    os.makedirs(ROOT, mode=0o700, exist_ok=True)
    with open(os.path.join(ROOT, "lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # the previous hour's run is still at it
        st = load_state()
        st.setdefault("notes", [])
        step(st, datetime.datetime.now(), a.now, a.base, a.hold)
        try:
            flush_notes(st)
        finally:
            save_state(st)


if __name__ == "__main__":
    main()
