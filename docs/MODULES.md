# Code-to-definition map

| Mathematical object / operation | Implementation |
|---|---|
| Binary matrix rank over GF(2) | `states.rank_gf2` |
| Closed R-hop ball | `states.balls` |
| Global and rooted KnotStates | `states.extract_order` |
| Exact induced-subset counts | `states.exact_histogram` |
| Hybrid sampled estimates | `states.sampled_histogram` |
| Exact/estimated metadata | `g`, `l`, `g_hat`, `l_hat`, `g_exact`, `l_exact` |
| Cumulative Local+Global feature input | `states.feature_matrix` |
| GIN neighbor sum and graph pooling | `model.KSGL.forward` |
| Model order expansion | `model.KSGL.expand_order` |
| Adam-state expansion | `model.expand_adam_state` |
| Exact rooted joint multiset | `screen.node_keys` |
| KS-initialized symbolic refinement | `screen.refine_keys` |
| Global/local pair discrepancy | `screen.pair_stats` |
| Progressive SR collision classes | `screen.run_screen` |
| BREC neural training | `train.run_brec` |
| Shared SR neural encoder | `train.run_sr` |
| Checkpoint and RNG restoration | `train.initialize`, `train.save_training` |
| Atomic output / run locks | `runtime.py` |
| Portable public benchmark bootstrap | `scripts/setup_data.py` |
| Existing-file content locator | `scripts/locate_data.py` |

## Independent controls

KSGL keeps three controls separate:

- `order i`: maximum induced-subgraph size included in the cumulative KnotState representation;
- `radius R`: closed graph neighborhood used by the rooted/local KnotState;
- `layers T`: GIN propagation depth.

At order `i`, the feature vector contains every Local and Global block from `1` through `i`. With all nullity bins retained, the per-node input dimension is

`1 + i(i+3)`.

For example, order 6 has dimension 55 and order 15 has dimension 271 and order 20 has dimension 461.

## Exactness and high order

Orders `1..6` use a precomputed nullity lookup table for speed. Orders above 6 use exact GF(2) elimination on each induced principal submatrix. The mathematical definition is unchanged.

The exact screen never turns a missing coordinate into equality. If a population exceeds `exact_limit`, that coordinate is `unknown`. Hybrid sampling is available for neural experiments but is not inserted into the exact invariant hash.
