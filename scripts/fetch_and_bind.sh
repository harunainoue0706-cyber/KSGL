#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
exec "$PY" "$ROOT/scripts/setup_data.py" --data-dir "$ROOT/data/benchmarks" --manifest "$ROOT/configs/datasets.local.json" "$@"
