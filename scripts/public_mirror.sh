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
# the old side of each rule (regex rules: the words they remove) must be gone
sed -n 's/==>.*//p' "$RULES/blobs.txt" "$RULES/messages.txt" | sed 's/^regex:.*(\(.*\))\\n$/\1/' | tr '|' '\n' |
  sed 's/^[a-z]*\\t//' | while IFS= read -r old; do
    if grep -qF -- "$old" "$T/log"; then echo "FAIL: still there: $old"; touch "$T/fail"; fi
  done
for p in gmail MacBook-Air .agent-deck/worktree; do
  if grep -q -- "$p" "$T/log"; then echo "FAIL: $p"; touch "$T/fail"; fi
done
[ ! -e "$T/fail" ] || { rm -rf "$T"; exit 1; }
rm -rf "$T"
echo "checked: ok"
