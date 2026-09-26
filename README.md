\# KSGL: Algebraic Knot Deep Graph Learning



This repository contains the anonymous implementation accompanying the manuscript \*\*Algebraic Knot Deep Graph Learning for Structural Generalization\*\*.



KSGL augments graph learning with multi-order local and global \*\*KnotState\*\* representations. For an induced vertex set \\(S\\), the KnotState is the nullity of the induced adjacency matrix over \\(\\mathbb F\_2\\):



\\\[

\\nu\_2(S)=|S|-\\operatorname{rank}\_{\\mathbb F\_2} A\_G\[S].

\\]



For induced order \\(s\\), KSGL uses both a graph-level distribution



\\\[

H\_G^s(r)=\\#\\{S\\subseteq V:|S|=s,\\nu\_2(S)=r\\},

\\]



and a rooted local distribution



\\\[

H\_{G,u}^{s,R}(r)=\\#\\{S\\subseteq B\_R(u):u\\in S,|S|=s,\\nu\_2(S)=r\\}.

\\]





The maximum order \\(K\\) is cumulative: all Local and Global blocks from orders \\(1\\)

through \\(K\\) are retained. The resulting node representation is then processed by graph refinement. In the structural-resolution experiments, this is implemented as a

deterministic 1-WL/GIN-style refinement; in the neural experiments, it is

implemented by a trainable GIN encoder.



For the structural-resolution experiments reported in Tables 1 and 2 of the paper,

we use the deterministic KSGL refinement, denoted `ks\_refinement` in the output

files. It combines the cumulative Local and Global KnotState representations with

a deterministic 1-WL/GIN-style graph refinement. No additional supervised classifier

is trained for graph-pair discrimination.





\## Repository layout



```text

KSGL/

├── ksgl/                    # KnotState extraction, screening, and neural model

├── configs/                 # benchmark registry

├── scripts/                 # data setup and benchmark runners

├── tests/                   # regression/unit tests

├── validation/              # compact SRG(25,12,5,6) validation data

├── docs/                    # mathematical/code mapping and paper reproduction

└── experiments/

&#x20;   └── zeolite/             # zeolite interpolation/generalization experiments

```



\## Installation



Core structural experiments require Python 3.9+:



```bash

python -m pip install -r requirements.txt

python -m unittest discover -s tests -v

```



The test suite covers exact GF(2) rank/nullity extraction, permutation equivariance, low-order closed forms, cumulative order expansion, checkpoint recovery, benchmark discovery, and the exact high-order fallback through \\(K=20\\).



\## Minimal exact structural check



A complete 15-graph `SRG(25,12,5,6)` collection is included for a self-contained check:



```bash

python -u -m ksgl screen \\

&#x20; --data validation/sr251256.g6 \\

&#x20; --kind sr \\

&#x20; --srg 25,12,5,6 \\

&#x20; --orders 1,2,3,4,5,6 \\

&#x20; --radius 1 \\

&#x20; --workers 4 \\

&#x20; --exact-limit 2000000 \\

&#x20; --out results/SR25\_exact \\

&#x20; --cache results/SR25\_cache

```



Expected compact summaries are stored in `validation/`.



\## Benchmark data setup



Large graph benchmark collections are not duplicated in the core package. The public data bootstrap is:



```bash

python scripts/setup\_data.py

```



It creates `data/benchmarks/`, validates graph counts and strongly-regular parameters, and writes:



```text

configs/datasets.local.json

```



If the benchmark files already exist locally, use:



```bash

python scripts/locate\_data.py \\

&#x20; --root /path/to/data \\

&#x20; --out configs/datasets.local.json

```



\## Reproducing Tables 1--2: KSGL structural resolution



For the order-dependent structural-resolution experiment, KSGL uses the

cumulative Local and Global KnotState representations up to order \\(K\\),

followed by GIN/1-WL-equivalent message-passing refinement. No additional

supervised classifier is trained for graph-pair discrimination.



```bash

python -u scripts/run\_suite.py \\

&#x20; --manifest configs/datasets.local.json \\

&#x20; --names paper-table \\

&#x20; --stage screen \\

&#x20; --orders 1,2,3,4,5,6,7,8,9,10 \\

&#x20; --radius 1 \\

&#x20; --workers 12 \\

&#x20; --exact-limit 100000000 \\

&#x20; --out results/SR\_ORDER\_SWEEP

```



The screen reports four deterministic views:



```text

global

local

joint

ks\_refinement

```



Their meanings are:



\- `global`: cumulative global KnotState representation;

\- `local`: cumulative rooted local KnotState representation;

\- `joint`: combined Local + Global KnotState representation;

\- `ks\_refinement`: the deterministic KSGL structural refinement obtained

&#x20; by applying 1-WL/GIN-style graph refinement to the KnotState-augmented

&#x20; node representations.



The `ks\_refinement` results correspond to the KSGL results reported in

Tables 1 and 2 of the manuscript. The other three views are used for the

component ablation in Table 2.



Two non-isomorphic graphs are counted as separated whenever their resulting

KSGL structural representations differ.



\## Higher-order experiments



The public interface accepts orders through \\(K=20\\):



```bash

python -u scripts/run\_suite.py \\

&#x20; --manifest configs/datasets.local.json \\

&#x20; --names SRG36\_14 \\

&#x20; --stage screen \\

&#x20; --orders 1,2,3,4,5,6,7,8,9,10 \\

&#x20; --radius 1 \\

&#x20; --workers 12 \\

&#x20; --exact-limit 100000000 \\

&#x20; --out results/SRG36\_14

```



High-order exact enumeration grows combinatorially. A coordinate whose exact population exceeds `--exact-limit` is recorded as `unknown`; it is never silently treated as equality.



\## BREC



After data setup, exact structural screening can be launched with:



```bash

python -u scripts/run\_suite.py \\

&#x20; --manifest configs/datasets.local.json \\

&#x20; --names BREC \\

&#x20; --stage screen \\

&#x20; --orders 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \\

&#x20; --radius 1 \\

&#x20; --workers 12 \\

&#x20; --exact-limit 100000000 \\

&#x20; --out results/BREC\_ORDER\_SWEEP

```



The neural BREC protocol is a separate experiment:



```bash

python -u scripts/run\_suite.py \\

&#x20; --manifest configs/datasets.local.json \\

&#x20; --names BREC \\

&#x20; --stage neural \\

&#x20; --orders 4,6,10,15,20 \\

&#x20; --radius 1 \\

&#x20; --workers 8 \\

&#x20; --epochs 100 \\

&#x20; --device auto \\

&#x20; --out results/BREC\_NEURAL

```



\## Trained neural KSGL on SR graphs (separate protocol)



This experiment trains the parameterized KSGL encoder and is separate from

the deterministic structural-resolution protocol used for Tables 1 and 2.



```bash

python -u -m ksgl train-sr \\

&#x20; --data validation/sr251256.g6 \\

&#x20; --srg 25,12,5,6 \\

&#x20; --orders 1,2,3,4,5,6 \\

&#x20; --radius 1 \\

&#x20; --workers 4 \\

&#x20; --epochs 100 \\

&#x20; --device auto \\

&#x20; --out results/SR25\_neural \\

&#x20; --cache results/SR25\_cache

```



\## Zeolite property prediction



The zeolite code used for interpolation and unseen-topology generalization is in:



```text

experiments/zeolite/

```



It includes GNN, symmetry-informed GNN, KSGL, ALIGNN-style, and Matformer-style backbones under one shared data/training protocol. External experiment logging is disabled by default. See `experiments/zeolite/README.md`.



\## Mapping to the manuscript



A table-by-table command map is provided in:



```text

docs/PAPER\_REPRODUCTION.md

```



\## Reproducibility rules



\- Exact structural witnesses use integer KnotState counts only.

\- Missing exact coordinates are marked `unknown`.

\- Hybrid sampling is allowed only in explicitly selected neural/scalable experiments.

\- Induced order \\(K\\), rooted radius \\(R\\), and GNN propagation depth are independent controls.

\- Results, caches, checkpoints, run locks, and external logging artifacts are not tracked.



\## Anonymous-release note



This repository intentionally contains no author names, affiliations, personal e-mail addresses, machine-specific home paths, or citation metadata that identifies the submission team. A citation file can be added after de-anonymization.



