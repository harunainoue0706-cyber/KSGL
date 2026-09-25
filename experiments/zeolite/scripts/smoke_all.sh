#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
export PROJECT_ROOT="$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export WANDB_MODE=disabled
export SYMGNN_OFFLINE_GNN=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p smoke_logs
for MODEL in gnn symgnn ksgl alignn matformer; do
  case "$MODEL" in
    gnn|symgnn|ksgl) BATCH=128 ;;
    alignn) BATCH=4 ;;
    matformer) BATCH=8 ;;
  esac
  if [[ "$MODEL" == "ksgl" ]]; then export USE_KS_FEATURES=1; else export USE_KS_FEATURES=0; fi

  for TASK in gen int; do
    EXP="SMOKE_${TASK^^}_${MODEL}"
    LOG="smoke_logs/${EXP}.log"
    if [[ "$TASK" == "gen" ]]; then
      "$PY" -u -m symgraph.train \
        dataset=gen model="$MODEL" \
        'dataset.val_codes=[CHA,ITW]' 'dataset.test_codes=[CHA,ITW]' \
        dataset.val_split=0.1 dataset.test_split=0.1 \
        dataset.batch_size="$BATCH" dataset.n_samples=250 \
        dataset.edge_type=radius dataset.radius=8.0 dataset.seed=42 \
        +train_seed=2023 epochs=1 expname="$EXP" > "$LOG" 2>&1
    else
      "$PY" -u -m symgraph.train \
        dataset=int model="$MODEL" \
        dataset.batch_size="$BATCH" dataset.n_samples=250 \
        dataset.edge_type=radius dataset.radius=8.0 dataset.seed=42 \
        +train_seed=2023 epochs=1 expname="$EXP" > "$LOG" 2>&1
    fi
    tail -n 12 "$LOG"
  done
done

echo "ALL SMOKE TESTS FINISHED"
