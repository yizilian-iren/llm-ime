#!/bin/sh
# Runs on dgx next to training: kill our training if available memory drops below the floor (GB).
# dgx memory is unified; running it out freezes the whole shared machine.
FLOOR=${1:-25}
while pgrep -f scripts/train.py >/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$FLOOR" ]; then
    pkill -9 -f scripts/train.py
    echo "$(date '+%F %T') killed train.py: MemAvailable ${avail} GB < ${FLOOR} GB"
  fi
  sleep 1
done
