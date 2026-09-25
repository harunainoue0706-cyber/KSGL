#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash scripts/run_one.sh {gnn|symgnn|ksgl|alignn|matformer} {gen|int} {2023|2024|2025}"
  exit 2
fi

MODEL="$1"
TASK="$2"
SEED="$3"

case "$MODEL" in
  gnn|symgnn|ksgl) BATCH=128 ;;
  alignn)           BATCH=4   ;;
  matformer)        BATCH=8   ;;
  *) echo "Unknown model: $MODEL"; exit 2 ;;
esac

case "$TASK" in
  gen|int) ;;
  *) echo "Unknown task: $TASK"; exit 2 ;;
esac

case "$SEED" in
  2023|2024|2025) ;;
  *) echo "Unknown seed: $SEED"; exit 2 ;;
esac

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
export PYTHONHASHSEED="$SEED"

if [[ "$MODEL" == "ksgl" ]]; then
  export USE_KS_FEATURES=1
else
  export USE_KS_FEATURES=0
fi

mkdir -p logs
EXP="${TASK^^}_${MODEL}_seed${SEED}"
LOG="logs/${EXP}.log"

COMMON=(
  model="$MODEL"
  dataset.batch_size="$BATCH"
  dataset.n_samples=250
  dataset.edge_type=radius
  dataset.radius=8.0
  dataset.seed=42
  +train_seed="$SEED"
  epochs=400
  expname="$EXP"
)

if [[ "$TASK" == "gen" ]]; then
  "$PY" -u -m symgraph.train \
    dataset=gen \
    'dataset.val_codes=[CHA,ITW]' \
    'dataset.test_codes=[CHA,ITW]' \
    dataset.val_split=0.1 \
    dataset.test_split=0.1 \
    "${COMMON[@]}" \
    2>&1 | tee "$LOG"
else
  "$PY" -u -m symgraph.train \
    dataset=int \
    "${COMMON[@]}" \
    2>&1 | tee "$LOG"
fi
