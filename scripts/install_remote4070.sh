#!/bin/sh
# The engine's remote half on home4070 (7E.1, ime.remote): installs it from HEAD, from M4. Safe to run again.
#   scripts/install_remote4070.sh          the service on home4070 (code, models, venv, systemd user service)
#   scripts/install_remote4070.sh m3       and M3's remote.json (its engine service then asks home4070 first;
#                                          restart it, or deploy, to take it up)
#   scripts/install_remote4070.sh off      removes M3's remote.json (all on M3 again; the service stays)
#
# On home4070 (WSL, MLX's CUDA backend):
#   ~/llm-ime/.venv-mlx/                   python 3.12, mlx[cuda], mlx-lm, numpy, jieba, pypinyin, CUDA headers
#   ~/llm-ime/cuda-home/include/           the pip CUDA packages' headers in one place (CUDA_HOME: MLX compiles
#                                          kernels at run time, cached on disk)
#   ~/llm-ime/remote/                      this commit's ime/ and scripts/ (git archive), models/
#   ~/.config/llm-ime/remote-token         the token M3 says hello with (600)
#   systemd user service llm-ime-remote    python -m ime.remote serve on the Tailscale address only
# Its log (journalctl --user -u llm-ime-remote) has times and errors only; nothing typed is written there (D-049).
# The models: r6-4bit and the base mixed in, and the personal model M3 runs (models/personal-4bit there), from
# M4's copy (the weekly retrain's held/ or runs/); scripts/retrain.py pushes each new one after deploying it to M3.
set -e
cd "$(dirname "$0")/.."
SSH="ssh -o ClearAllForwardings=yes"
GPU=home4070
TS_IP=100.100.59.15  # home4070's Tailscale address (the WSL side)
PORT=8771
MODELS="r6-4bit qwen3-0.6b-base-4bit"
RETRAIN="$HOME/Library/Application Support/llm-ime-retrain"
M3_CONF='$HOME/Library/Application Support/llm-ime/remote.json'

if [ "$1" = off ]; then
  $SSH m3 "rm -f \"$M3_CONF\"" && echo "M3: remote.json removed (restart the engine service to take it up)"
  exit 0
fi

git diff --quiet HEAD -- ime scripts || { echo "commit ime/ scripts/ first: the service is installed from HEAD" >&2; exit 1; }

echo "== venv"
$SSH $GPU 'set -e; cd ~/llm-ime
  [ -x .venv-mlx/bin/python ] || ~/.local/bin/uv venv -q --python 3.12 .venv-mlx
  VIRTUAL_ENV=.venv-mlx ~/.local/bin/uv pip install -q "mlx[cuda]==0.32.3" "mlx-lm==0.32.0" numpy jieba==0.42.1 \
    pypinyin==0.55.0 "nvidia-cuda-runtime-cu12==12.9.*" "nvidia-cuda-cccl-cu12==12.9.*"
  N=$PWD/.venv-mlx/lib/python3.12/site-packages/nvidia; mkdir -p cuda-home/include
  for d in cuda_runtime cuda_cccl cuda_nvrtc; do for f in $N/$d/include/*; do ln -sfn $f cuda-home/include/; done; done'

echo "== models"
$SSH $GPU 'mkdir -p ~/llm-ime/remote/models'
for m in $MODELS; do
  rsync -a --partial -e "$SSH" "models/$m/" "$GPU:llm-ime/remote/models/$m/"
done
P=$($SSH m3 'readlink "$HOME/Projects/llm-ime/models/personal-4bit" || true')
if [ -n "$P" ]; then
  SRC=$(ls -d "$RETRAIN/held/$P" "$RETRAIN"/runs/*/"$P" 2>/dev/null | head -1)
  if [ -n "$SRC" ]; then
    rsync -a --delete -e "$SSH" "$SRC/" "$GPU:llm-ime/remote/models/$P/"
    $SSH $GPU "cd ~/llm-ime/remote/models && ln -sfn $P personal-4bit && for d in personal-2*-4bit; do [ \$d = $P ] || rm -rf \$d; done"
  else
    echo "no copy of M3's $P on M4: home4070 runs the base, and M3 will not use it until the next retrain" >&2
  fi
else
  $SSH $GPU 'cd ~/llm-ime/remote/models && rm -rf personal-4bit personal-2*-4bit'
fi

echo "== code ($(git rev-parse --short HEAD))"
git archive --format=tar HEAD ime scripts | $SSH $GPU "set -e; R=~/llm-ime/remote
  rm -rf \$R/ime \$R/scripts; tar -x -C \$R; echo $(git rev-parse HEAD) > \$R/COMMIT"

echo "== token and service"
$SSH $GPU "set -e
  mkdir -p -m 700 ~/.config/llm-ime
  [ -s ~/.config/llm-ime/remote-token ] || (umask 077; head -c 24 /dev/urandom | base64 > ~/.config/llm-ime/remote-token)
  mkdir -p ~/.config/systemd/user
  cat > ~/.config/systemd/user/llm-ime-remote.service <<EOF
[Unit]
Description=llm-ime engine, remote half for M3 (7E.1)
After=network-online.target

[Service]
WorkingDirectory=%h/llm-ime/remote
Environment=CUDA_HOME=%h/llm-ime/cuda-home MLX_CUDA_SDPA_CACHE_SIZE=8192 MLX_CUDA_GRAPH_CACHE_SIZE=4096 MLX_ENABLE_CACHE_THRASHING_CHECK=0 PYTHONUNBUFFERED=1
ExecStart=%h/llm-ime/.venv-mlx/bin/python -W ignore -m ime.remote serve --host $TS_IP --port $PORT
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable -q llm-ime-remote
  systemctl --user restart llm-ime-remote"

echo "== check"
for i in $(seq 1 90); do
  $SSH $GPU 'journalctl --user -u llm-ime-remote --since "-3min" -o cat | grep -q "listening on"' && break
  [ $i = 90 ] && { $SSH $GPU 'journalctl --user -u llm-ime-remote -n 30 -o cat'; exit 1; }
  sleep 2
done
$SSH $GPU 'journalctl --user -u llm-ime-remote --since "-3min" -o cat | grep "loaded\|listening" | tail -2'

if [ "$1" = m3 ]; then
  TOKEN=$($SSH $GPU 'cat ~/.config/llm-ime/remote-token')
  printf '{"host": "%s", "port": %s, "token": "%s"}\n' "$TS_IP" "$PORT" "$TOKEN" | \
    $SSH m3 "umask 077; cat > \"$M3_CONF\"" && echo "M3: remote.json written (restart the engine service to take it up)"
fi
