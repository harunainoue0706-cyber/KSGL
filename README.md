# KSGL: Algebraic Knot Deep Graph Learning

This repository contains the anonymous implementation accompanying the manuscript **Algebraic Knot Deep Graph Learning for Structural Generalization**.

KSGL augments graph learning with multi-order local and global **KnotState** representations. For an induced vertex set \(S\), the KnotState is the nullity of the induced adjacency matrix over \(\mathbb F_2\):

\[
\nu_2(S)=|S|-\operatorname{rank}_{\mathbb F_2} A_G[S].
\]

For induced order \(s\), KSGL uses both a graph-level distribution

\[
H_G^s(r)=\#\{S\subseteq V:|S|=s,\nu_2(S)=r\},
\]

and a rooted local distribution

\[
H_{G,u}^{s,R}(r)=\#\{S\subseteq B_R(u):u\in S,|S|=s,\nu_2(S)=r\}.
\]

The maximum order \(K\) is cumulative: all Local and Global blocks from orders \(1\) through \(K\) are retained. The resulting node representation is passed to a shared GNN.

## Repository layout

```text
KSGL/
├── ksgl/                    # KnotState extraction, screening, and neural model
├── configs/                 # benchmark registry
├── scripts/                 # data setup and benchmark runners
├── tests/                   # regression/unit tests
├── validation/              # compact SRG(25,12,5,6) validation data
├── docs/                    # mathematical/code mapping and paper reproduction
└── experiments/
    └── zeolite/             # zeolite interpolation/generalization experiments
```

## Installation

Core structural experiments require Python 3.9+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The test suite covers exact GF(2) rank/nullity extraction, permutation equivariance, low-order closed forms, cumulative order expansion, checkpoint recovery, benchmark discovery, and the exact high-order fallback through \(K=20\).

## Minimal exact structural check

A complete 15-graph `SRG(25,12,5,6)` collection is included for a self-contained check:

```bash
python -u -m ksgl screen \
  --data validation/sr251256.g6 \
  --kind sr \
  --srg 25,12,5,6 \
  --orders 1,2,3,4,5,6 \
  --radius 1 \
  --workers 4 \
  --exact-limit 2000000 \
  --out results/SR25_exact \
  --cache results/SR25_cache
```

Expected compact summaries are stored in `validation/`.

## Benchmark data setup

Large graph benchmark collections are not duplicated in the core package. The public data bootstrap is:

```bash
python scripts/setup_data.py
```

It creates `data/benchmarks/`, validates graph counts and strongly-regular parameters, and writes:

```text
configs/datasets.local.json
```

If the benchmark files already exist locally, use:

```bash
python scripts/locate_data.py \
  --root /path/to/data \
  --out configs/datasets.local.json
```

## Order-dependent SR-family screen

```bash
python -u scripts/run_suite.py \
  --manifest configs/datasets.local.json \
  --names paper-table \
  --stage screen \
  --orders 1,2,3,4,5,6,7,8,9,10 \
  --radius 1 \
  --workers 12 \
  --exact-limit 100000000 \
  --out results/SR_ORDER_SWEEP
```

The screen reports four deterministic views:

```text
global
local
joint
ks_refinement
```

`ks_refinement` is the KnotState-initialized structural refinement used for the cumulative pair-separation analysis. It is deterministic and is reported separately from trained neural evaluation.

## Higher-order experiments

The public interface accepts orders through \(K=20\):

```bash
python -u scripts/run_suite.py \
  --manifest configs/datasets.local.json \
  --names SRG36_14 \
  --stage screen \
  --orders 1,2,3,4,5,6,7,8,9,10 \
  --radius 1 \
  --workers 12 \
  --exact-limit 100000000 \
  --out results/SRG36_14
```

High-order exact enumeration grows combinatorially. A coordinate whose exact population exceeds `--exact-limit` is recorded as `unknown`; it is never silently treated as equality.

## BREC

After data setup, exact structural screening can be launched with:

```bash
python -u scripts/run_suite.py \
  --manifest configs/datasets.local.json \
  --names BREC \
  --stage screen \
  --orders 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --radius 1 \
  --workers 12 \
  --exact-limit 100000000 \
  --out results/BREC_ORDER_SWEEP
```

The neural BREC protocol is a separate experiment:

```bash
python -u scripts/run_suite.py \
  --manifest configs/datasets.local.json \
  --names BREC \
  --stage neural \
  --orders 4,6,10,15,20 \
  --radius 1 \
  --workers 8 \
  --epochs 100 \
  --device auto \
  --out results/BREC_NEURAL
```

## Neural KSGL on SR graphs

```bash
python -u -m ksgl train-sr \
  --data validation/sr251256.g6 \
  --srg 25,12,5,6 \
  --orders 1,2,3,4,5,6 \
  --radius 1 \
  --workers 4 \
  --epochs 100 \
  --device auto \
  --out results/SR25_neural \
  --cache results/SR25_cache
```

## Zeolite property prediction

The zeolite code used for interpolation and unseen-topology generalization is in:

```text
experiments/zeolite/
```

It includes GNN, symmetry-informed GNN, KSGL, ALIGNN-style, and Matformer-style backbones under one shared data/training protocol. External experiment logging is disabled by default. See `experiments/zeolite/README.md`.

## Mapping to the manuscript

A table-by-table command map is provided in:

```text
docs/PAPER_REPRODUCTION.md
```

## Reproducibility rules

- Exact structural witnesses use integer KnotState counts only.
- Missing exact coordinates are marked `unknown`.
- Hybrid sampling is allowed only in explicitly selected neural/scalable experiments.
- Induced order \(K\), rooted radius \(R\), and GNN propagation depth are independent controls.
- Results, caches, checkpoints, run locks, and external logging artifacts are not tracked.

## Anonymous-release note

This repository intentionally contains no author names, affiliations, personal e-mail addresses, machine-specific home paths, or citation metadata that identifies the submission team. A citation file can be added after de-anonymization.
