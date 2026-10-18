#!/bin/sh
# Installs the weekly retrain (6A.2) on M4, the Mac it runs on, from HEAD. Safe to run again.
#   scripts/install_retrain.sh            install, or update the code
#   scripts/install_retrain.sh --remove   stop it (the runs' numbers in runs/ stay)
#
# ~/Library/Application Support/llm-ime-retrain/
#   code/          this commit's ime/ and scripts/ (git archive); models, data, .venv -> ~/Projects/llm-ime
#   state.json     the week's run (scripts/retrain.py); runs/<day>/ its numbers and adapter; work/ while it runs
# ~/Library/LaunchAgents/com.llm-ime.retrain.plist   scripts/retrain.py every hour
# ~/Library/Logs/llm-ime/retrain.log                 what it did (counts and times only)
# Needs on M4: ~/Projects/llm-ime/models/<base>-hf for the deployed base, and data/private/eval.jsonl; on home4070:
# ~/llm-ime/.venv-train, data/replay/<base>.npz (public samples in the base's format; 6A.1).
set -e
cd "$(dirname "$0")/.."
ROOT="$HOME/Library/Application Support/llm-ime-retrain"
REPO="$HOME/Projects/llm-ime"
LABEL=com.llm-ime.retrain
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
D=gui/$(id -u)

launchctl bootout $D/$LABEL 2>/dev/null || true
if [ "$1" = --remove ]; then rm -f "$PLIST"; echo "removed $LABEL"; exit 0; fi

git diff --quiet HEAD -- ime scripts || { echo "commit ime/ scripts/ first: the retrain runs HEAD's code" >&2; exit 1; }
mkdir -p "$ROOT" "$HOME/Library/Logs/llm-ime" "$HOME/Library/LaunchAgents"
chmod 700 "$ROOT"
rm -rf "$ROOT/code.new"; mkdir "$ROOT/code.new"
git archive --format=tar HEAD ime scripts | tar -x -C "$ROOT/code.new"
for x in models data .venv; do ln -s "$REPO/$x" "$ROOT/code.new/$x"; done
git rev-parse HEAD > "$ROOT/code.new/COMMIT"
rm -rf "$ROOT/code.old"; [ -d "$ROOT/code" ] && mv "$ROOT/code" "$ROOT/code.old"
mv "$ROOT/code.new" "$ROOT/code"; rm -rf "$ROOT/code.old"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$ROOT/code/.venv/bin/python</string>
        <string>$ROOT/code/scripts/retrain.py</string>
    </array>
    <key>WorkingDirectory</key><string>$ROOT/code</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>PYTHONUNBUFFERED</key><string>1</string>
    </dict>
    <key>StartInterval</key><integer>3600</integer>
    <key>ProcessType</key><string>Background</string>
    <key>StandardOutPath</key><string>$HOME/Library/Logs/llm-ime/retrain.log</string>
    <key>StandardErrorPath</key><string>$HOME/Library/Logs/llm-ime/retrain.log</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null
launchctl bootstrap $D "$PLIST"
echo "installed $LABEL at $(git rev-parse --short HEAD): every hour; log ~/Library/Logs/llm-ime/retrain.log"
