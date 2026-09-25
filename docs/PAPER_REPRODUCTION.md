# Paper reproduction map

This file maps the anonymous code release to the experiments reported in the manuscript.

## Tables 1--2: order-dependent structural resolution

Set up the public graph collections:

```bash
python scripts/setup_data.py
```

Then run the cumulative order sweep:

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

The principal summary is `order_summary.csv`. The pair-separation tables use the cumulative KnotState structural-refinement view.

## Table 3: Global / Local / Joint / KSGL ablation

The exact screen writes separate rows for:

```text
global
local
joint
ks_refinement
```

Use `global`, `local`, and `joint` for the component ablation. The final KSGL column in the manuscript corresponds to the full KnotState-initialized structural-refinement view used by the order-dependent screening protocol.

## Table 4: non-isomorphic graph discrimination

The SR-family structural screen is produced by the same exact runner. The BREC pipeline is registered under the same manifest and can be run through maximum order 20.

Example exact BREC screen:

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

Neural BREC evaluation is intentionally separate from exact structural screening.

## Table 5: zeolite property prediction

See:

```text
experiments/zeolite/README.md
```

The shared protocol covers:

- interpolation with within-topology 80/10/10 splits;
- unseen-topology generalization with the held-out topologies used by the manuscript;
- three training seeds: 2023, 2024, 2025;
- 400 epochs;
- the shared HOA/isotherm objectives and evaluation metrics.

## Table 6: computational efficiency

The core KSGL model and the zeolite benchmark backbones are included here. The manuscript's unified cross-method efficiency table also depends on external baseline implementations that are not vendored into this anonymous repository. The release therefore does not claim one-command reproduction of every external baseline's memory/latency number.
