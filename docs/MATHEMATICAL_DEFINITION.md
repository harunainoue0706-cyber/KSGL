# Mathematical definition used by the code

For a simple undirected graph `G=(V,E)` and `S subseteq V`, let `A_G[S]` be the principal adjacency matrix of the induced subgraph.

The KnotState is

`nu_2(S) = |S| - rank_{F_2}(A_G[S])`.

At induced order `s`:

- Global: `H_G^s(r)` counts all `s`-vertex subsets with nullity `r`.
- Rooted local: `H_{G,u}^{s,R}(r)` counts all such subsets contained in the closed radius-`R` ball and containing root `u`.

At maximum order `i`, the representation is cumulative and contains every Global and Local block for `s=1,...,i`.

The code distinguishes three independent controls:

- induced order `i`,
- neighborhood radius `R`,
- GIN propagation depth `T`.

They are not interchangeable.
