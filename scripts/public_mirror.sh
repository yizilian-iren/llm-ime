#!/bin/sh
# Builds the public mirror of this repo (6C.1, D-040) in a new folder; this repo and its history are not touched.
#   scripts/public_mirror.sh RULES OUT [REF]      REF default main; OUT must not exist
#
# A fresh clone of REF only, rewritten by git-filter-repo with RULES/ (kept out of git, data/private/public-mirror):
#   mailmap       every author / committer to the GitHub noreply address
#   blobs.txt     --replace-text over every file of every commit (personal data out)
#   messages.txt  --replace-message over every commit message
# and without .agent-deck/. Then it checks the result: no author or committer but the noreply address, and none of
# the replaced texts left anywhere in `git log -p`. Pushing it is a separate step, after the user said yes.
set -e
RULES=$1 OUT=$2 REF=${3:-main}
[ -n "$RULES" ] && [ -n "$OUT" ] || { sed -n '2,3p' "$0"; exit 2; }
[ ! -e "$OUT" ] || { echo "$OUT exists"; exit 1; }
FR=${FILTER_REPO:-git-filter-repo}
SRC=$(git -C "$(dirname "$0")/.." rev-parse --path-format=absolute --git-common-dir)

git clone -q --no-local --single-branch --branch "$REF" "$SRC" "$OUT"
cd "$OUT"
git branch -m "$REF" main 2>/dev/null || true
"$FR" --force --quiet --mailmap "$RULES/mailmap" --replace-text "$RULES/blobs.txt" \
  --replace-message "$RULES/messages.txt" --invert-paths --path .agent-deck/

echo "commits: $(git rev-list --count main)"
emails=$(git log --format='%ae%n%ce' main | sort -u)
echo "addresses: $(echo $emails)"
bad=$(echo "$emails" | grep -v -e '+yizilian-iren@users.noreply.github.com$' || true)
[ -z "$bad" ] || { echo "FAIL: other addresses: $bad"; exit 1; }
T=$(mktemp -d)
git log -p --all --format='%an %ae %cn %ce%n%B' -- . ':!scripts/public_mirror.sh' > "$T/log"  # it names the patterns
# the old side of each rule must be gone: a plain rule as the text, a regex rule as the same regex (the whole line it
# removes, not its words: 焦虑 is in the public word lists). `git log -p` puts a marker before each line, so look at both
if ! python3 - "$T/log" "$RULES/blobs.txt" "$RULES/messages.txt" <<'PY'
import re, sys
log = open(sys.argv[1], encoding="utf-8", errors="replace").read()
lines = log.split("\n")
text = log + "\n" + "\n".join(l[1:] for l in lines)
bad = False
for path in sys.argv[2:]:
    for rule in open(path, encoding="utf-8").read().split("\n"):
        if "==>" not in rule:
            continue
        old = rule.split("==>")[0]
        if old.startswith("regex:"):
            hit = re.search(old[6:], text)
        else:
            hit = old and old in text
        if hit:
            print("FAIL: still there: " + (hit.group(0).strip() if hasattr(hit, "group") else old)); bad = True
sys.exit(1 if bad else 0)
PY
then touch "$T/fail"; fi
for p in gmail MacBook-Air .agent-deck/worktree; do
  if grep -q -- "$p" "$T/log"; then echo "FAIL: $p"; touch "$T/fail"; fi
done
[ ! -e "$T/fail" ] || { rm -rf "$T"; exit 1; }
rm -rf "$T"
echo "checked: ok"
