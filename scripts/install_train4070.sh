#!/bin/sh
# home4070's training environment (6A.1, scripts/retrain.py): ~/llm-ime/.venv-train from scripts/requirements-train.txt
# (python 3.13, torch cu130). Run from M4 after a fresh install; safe to run again. What else a fresh machine needs
# for the weekly retrain: models/<base>-hf (rsync from M4's models/) and data/replay/<base>.npz (from dgx,
# ~/llm-ime/data/replay-r6/train.npz is r6-4bit.npz: sha256 313dd9cf..., 268654906 B), then scripts/install_remote4070.sh.
set -e
cd "$(dirname "$0")/.."
SSH="ssh -o ClearAllForwardings=yes"
GPU=home4070
$SSH $GPU 'mkdir -p ~/llm-ime'
rsync -a -e "$SSH" scripts/requirements-train.txt "$GPU:llm-ime/requirements-train.txt"
$SSH $GPU 'set -e; cd ~/llm-ime
  [ -x .venv-train/bin/python ] || ~/.local/bin/uv venv -q --python 3.13 .venv-train
  VIRTUAL_ENV=.venv-train ~/.local/bin/uv pip install -q -r requirements-train.txt
  .venv-train/bin/python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"'
