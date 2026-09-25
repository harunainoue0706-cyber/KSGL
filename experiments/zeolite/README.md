# Zeolite property-prediction experiments

This directory contains the anonymous zeolite experiment code used for the manuscript's interpolation and unseen-topology generalization study.

## Tasks

- `gen`: unseen-topology generalization; the held-out topology codes are `CHA` and `ITW`.
- `int`: interpolation; structures are split within each retained topology with an 80/10/10 train/validation/test protocol.

The data split seed is fixed at `42`. Training randomness is evaluated with seeds:

```text
2023, 2024, 2025
```

Each full run uses 400 epochs.

## Models

The shared training pipeline exposes:

```text
gnn
symgnn
ksgl
alignn
matformer
```

`ksgl` augments the non-symmetry GNN backbone with the KnotState features used by the scalable zeolite experiment. The scalable zeolite preprocessing uses induced orders 4, 5, and 6, rooted radius 3, deterministic capped local enumeration/sampling, and deterministic global sampling. Precomputed KnotState arrays are included in `Data_numpy/` for the packaged benchmark structures.

## Tested environment

The reference environment is listed in:

```text
ENVIRONMENT_REFERENCE.json
```

Check the major dependencies with:

```bash
python scripts/check_install.py
```

External experiment logging is disabled by default.

## Smoke tests

```bash
bash scripts/smoke_all.sh
```

## Single run

```bash
bash scripts/run_one.sh ksgl gen 2023
bash scripts/run_one.sh ksgl int 2023
bash scripts/run_one.sh gnn gen 2023
bash scripts/run_one.sh symgnn gen 2023
bash scripts/run_one.sh alignn gen 2023
bash scripts/run_one.sh matformer gen 2023
```

## Full three-seed benchmark

```bash
nohup bash scripts/run_all.sh > full_benchmark.log 2>&1 &
tail -f full_benchmark.log
```

After all runs finish:

```bash
python scripts/summarize_all.py
cat results_summary/summary_mean_sd_se.csv
```

The summary reports mean and sample standard deviation across the three training seeds for:

- heat of adsorption MAE/MSE;
- adsorption-isotherm MAE/MSE.

## Reproducibility notes

- The split seed remains fixed while the training seed changes.
- All models use the same data loader, task targets, optimizer family, learning-rate schedule, checkpoint selection rule, and evaluation functions.
- Batch sizes are model-specific to match the benchmark setup.
- `WANDB_MODE=disabled` is set by the run scripts and the default logging configuration is disabled.
- Generated `models/`, `logs/`, `smoke_logs/`, and `results_summary/` directories are not tracked.
