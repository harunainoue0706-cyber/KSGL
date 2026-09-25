#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo 'Usage: bash scripts/start_detached.sh LOGFILE COMMAND [ARGS...]' >&2
  exit 2
fi
log="$1"
shift
mkdir -p "$(dirname "$log")"
# No shell re-interpretation of user arguments; no kill or overwrite of checkpoints.
nohup "$@" >> "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "${log}.pid"
printf 'PID=%s\nLog=%s\nWatch: tail -f "%s"\n' "$pid" "$log" "$log"
