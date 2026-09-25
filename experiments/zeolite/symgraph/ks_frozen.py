# Scalable KnotState feature map used by the zeolite KSGL experiment.

import itertools
import math
import os
import random
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch

S_ORDERS = (4, 5, 6)
KS_DIM = 2 * sum(s + 1 for s in S_ORDERS)  # 36


def adjlist_from_edges(n, edge_index):
    if torch.is_tensor(edge_index):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index = np.asarray(edge_index)
    adj = [set() for _ in range(n)]
    for u, v in edge_index.T.tolist():
        if u != v:
            adj[int(u)].add(int(v))
    return adj


def gf2_rank_from_subset(subset, adj):
    """Rank over GF(2) of the adjacency submatrix induced by subset; s<=6."""
    nodes = list(subset)
    s = len(nodes)
    pos = {u: i for i, u in enumerate(nodes)}
    rows = []
    for u in nodes:
        bits = 0
        for v in adj[u]:
            j = pos.get(v)
            if j is not None:
                bits ^= (1 << j)
        rows.append(bits)

    rank = 0
    col = 0
    while col < s and rank < s:
        pivot = None
        for r in range(rank, s):
            if (rows[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(s):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1
        col += 1
    return rank


def ego_nodes(root, adj, radius):
    seen = {root}
    q = deque([(root, 0)])
    while q:
        u, d = q.popleft()
        if d >= radius:
            continue
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return sorted(seen)


def sample_combinations(pool, choose, cap, rng):
    total = math.comb(len(pool), choose) if len(pool) >= choose else 0
    if total == 0:
        return [], 0
    if total <= cap:
        return list(itertools.combinations(pool, choose)), total
    out = set()
    while len(out) < cap:
        out.add(tuple(sorted(rng.sample(pool, choose))))
    return list(out), total


def hist_for_subsets(subsets, total_count, s, adj, normalization="probability"):
    h = np.zeros(s + 1, dtype=np.float64)
    if len(subsets) == 0:
        return h
    for S in subsets:
        r = gf2_rank_from_subset(S, adj)
        nu = s - r
        h[nu] += 1.0
    if normalization == "probability":
        h /= float(len(subsets))
    else:
        h *= float(total_count) / float(len(subsets))
    return h


def compute_ks_features(n, edge_index, radius=3, local_cap=512,
                        global_samples=10000, seed=0,
                        normalization="probability"):
    """
    Scalable KnotState feature map used by the zeolite KSGL experiment:
      local rooted histograms for s=4,5,6
      || graph-global histograms for s=4,5,6
    Exact enumeration is used when a population is below its cap; otherwise
    deterministic sampling is used. Each histogram has bins nu=0,...,s, hence
    18 local + 18 global = 36 dimensions per node.
    """
    adj = adjlist_from_edges(n, edge_index)
    rng = random.Random(seed)

    global_parts = []
    all_nodes = list(range(n))
    for s in S_ORDERS:
        if n < s:
            global_parts.append(np.zeros(s + 1, dtype=np.float64))
            continue
        combos, total = sample_combinations(all_nodes, s, global_samples, rng)
        global_parts.append(hist_for_subsets(combos, total, s, adj, normalization))
    gks = np.concatenate(global_parts).astype(np.float32)

    node_ks = []
    for root in range(n):
        ego = ego_nodes(root, adj, radius)
        others = [u for u in ego if u != root]
        parts = []
        for s in S_ORDERS:
            choose = s - 1
            if len(others) < choose:
                parts.append(np.zeros(s + 1, dtype=np.float64))
                continue
            combos, total = sample_combinations(others, choose, local_cap, rng)
            rooted = [(root,) + tuple(c) for c in combos]
            parts.append(hist_for_subsets(rooted, total, s, adj, normalization))
        lks = np.concatenate(parts).astype(np.float32)
        node_ks.append(np.concatenate([lks, gks], axis=0))
    return np.stack(node_ks, axis=0).astype(np.float32)


def _stable_seed(zeo, base_seed=42):
    return int(base_seed + (zlib.crc32(str(zeo).encode("utf-8")) & 0x7fffffff))


def ks_cache_name(radius=3, local_cap=512, global_samples=10000,
                  normalization="probability"):
    norm = "prob" if normalization == "probability" else "count"
    return f"ks_r{radius}_l{local_cap}_g{global_samples}_{norm}.npy"


def load_or_compute_ks_features(zeo, n, edge_index, data_root,
                                radius=3, local_cap=512,
                                global_samples=10000,
                                normalization="probability",
                                base_seed=42):
    data_root = Path(data_root)
    cache = data_root / str(zeo) / ks_cache_name(
        radius, local_cap, global_samples, normalization
    )
    if cache.exists():
        x = np.load(cache)
        if x.shape == (int(n), KS_DIM):
            return x.astype(np.float32, copy=False)
        print(f"[KS] ignoring bad cache shape {cache}: {x.shape}")

    x = compute_ks_features(
        int(n), edge_index,
        radius=radius,
        local_cap=local_cap,
        global_samples=global_samples,
        seed=_stable_seed(zeo, base_seed),
        normalization=normalization,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    return x
