#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for MODEL in gnn symgnn ksgl alignn matformer; do
  for TASK in gen int; do
    for SEED in 2023 2024 2025; do
      bash scripts/run_one.sh "$MODEL" "$TASK" "$SEED"
    done
  done
done

python scripts/summarize_all.py
