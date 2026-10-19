#!/bin/sh
# Builds the input method on M4 and installs it and the engine service on M3 (4A.3). Safe to run again.
#   scripts/deploy_m3.sh [host]        (default host: m3)
#
# On M3:
#   ~/Library/Input Methods/LLMIME.app              the shell (shell/build.sh)
#   ~/Library/Application Support/llm-ime/engine/   this commit's ime/ and scripts/ (git archive), models -> repo
#   ~/Library/LaunchAgents/com.llm-ime.engine.plist the service, run with the repo's .venv
#   ~/Library/Logs/llm-ime/engine.log               its log (times and errors only)
#   ~/Library/LaunchAgents/com.llm-ime.summary.plist the typing log's summary, hourly (5A.2)
#   ~/Library/Application Support/llm-ime/log/      the typing log and its summaries (the shell writes it)
#   ~/Library/Application Support/llm-ime/words.json the user's word list (5A.3, the service writes it)
# The repo there (~/Projects/llm-ime) only lends its .venv and models; missing model files are copied in.
# M3's ssh is restricted (no pty, no forwarding): everything below is one non-interactive command each.
set -e
HOST=${1:-m3}
SSH="ssh -o ClearAllForwardings=yes $HOST"
cd "$(dirname "$0")/.."
REPO_M3='$HOME/Projects/llm-ime'
BASE_M3='$HOME/Library/Application Support/llm-ime'
LABEL=com.llm-ime.engine
SUMMARY=com.llm-ime.summary
MODELS="r6-4bitqwen3-0.6b-base-4bit"  # what engine v02 loads

git diff --quiet HEAD -- ime scripts shell || { echo "commit ime/ scripts/ shell/ first: the engine is deployed from HEAD" >&2; exit 1; }

echo "== build"
shell/build.sh

echo "== models"
for m in $MODELS; do
  rsync -a --partial "models/$m/" "$HOST:Projects/llm-ime/models/$m/"
done

echo "== engine code ($(git rev-parse --short HEAD))"
git archive --format=tar HEAD ime scripts | $SSH "set -e
  B=\"$BASE_M3\"; mkdir -p \"\$B\"; rm -rf \"\$B/engine.new\"; mkdir \"\$B/engine.new\"
  tar -x -C \"\$B/engine.new\"
  ln -s \"$REPO_M3/models\" \"\$B/engine.new/models\"
  echo $(git rev-parse HEAD) > \"\$B/engine.new/COMMIT\"
  rm -rf \"\$B/engine.old\"; [ -d \"\$B/engine\" ] && mv \"\$B/engine\" \"\$B/engine.old\"
  mv \"\$B/engine.new\" \"\$B/engine\"; rm -rf \"\$B/engine.old\""

echo "== input method"
tar -C shell/build -cf - LLMIME.app | $SSH "set -e
  D=\"\$HOME/Library/Input Methods\"; mkdir -p \"\$D\"; rm -rf \"\$D/LLMIME.app.new\"; mkdir \"\$D/LLMIME.app.new\"
  tar -x -C \"\$D/LLMIME.app.new\"
  rm -rf \"\$D/LLMIME.app\"; mv \"\$D/LLMIME.app.new/LLMIME.app\" \"\$D/LLMIME.app\"; rmdir \"\$D/LLMIME.app.new\"
  pkill -x LLMIME || true"  # macOS starts the new one the next time it is used

# install_agent LABEL TEMPLATE LOGNAME: fills in the template and (re)loads it as a LaunchAgent.
install_agent() {
  sed -e "s|@LABEL@|$1|" -e "s|@ARGS@||" "$2" | $SSH "set -e
  B=\"$BASE_M3\"; R=\"$REPO_M3\"; L=\"\$HOME/Library/Logs/llm-ime\"; mkdir -p \"\$L\" \"\$HOME/Library/LaunchAgents\"
  P=\"\$HOME/Library/LaunchAgents/$1.plist\"
  sed -e \"s|@REPO@/.venv|\$R/.venv|\" -e \"s|@REPO@|\$B/engine|\" -e \"s|@LOG@|\$L/$3|\" > \"\$P\"
  plutil -lint \"\$P\" >/dev/null
  D=gui/\$(id -u)
  launchctl bootout \$D/$1 2>/dev/null || true
  for i in \$(seq 1 40); do launchctl print \$D/$1 >/dev/null 2>&1 || break; sleep 0.25; done  # bootout returns before it is gone
  for i in 1 2 3 4 5; do launchctl bootstrap \$D \"\$P\" && break; [ \$i = 5 ] && exit 1; sleep 1; done"
}

echo "== service"
install_agent $LABEL scripts/llm-ime.engine.plist.in engine.log
install_agent $SUMMARY scripts/llm-ime.summary.plist.in summary.log

echo "== check"
$SSH "set -e
  B=\"$BASE_M3\"; S=\"\$B/engine.sock\"
  cd \"\$B/engine\"
  for i in \$(seq 1 60); do
    OUT=\$(\"$REPO_M3/.venv/bin/python\" scripts/engine_client.py nihao --timeout 20 2>/dev/null) && { echo \"\$OUT\" | tail -1; break; }
    [ \$i = 60 ] && { tail -20 \"\$HOME/Library/Logs/llm-ime/engine.log\"; exit 1; }
    sleep 1
  done
  echo \"engine \$(cut -c1-7 \"\$B/engine/COMMIT\"), input method \$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' \"\$HOME/Library/Input Methods/LLMIME.app/Contents/Info.plist\")\""
