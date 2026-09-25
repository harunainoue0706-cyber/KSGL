"""Benchmark backbones adapted to the shared zeolite HOA + isotherm task.

ALIGNN adapter:
  - follows the ALIGNN implementation already shipped with the SymGNN snapshot;
  - uses periodic edge displacement vectors supplied as ``data.edge_vec``;
  - exposes the same two-output interface as the existing GNN trainer.

Matformer adapter:
  - uses a PyG-style Matformer attention block
    commit 9fd30550829b313ff6a1bf819ebe6bcaaeb72b52;
  - uses the current benchmark's scalar atom input and radius graph;
  - replaces only the property head with the common HOA + isotherm head.

Both classes return ``(hoa_pred, q_prime_hat)`` so the dataset, objective,
pressure grid, optimizer, checkpoint selection, and evaluation remain shared.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_scatter import scatter_add, scatter_mean


class RBFExpansion(nn.Module):
    def __init__(self, vmin=0.0, vmax=8.0, bins=80, lengthscale=None):
        super().__init__()
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.bins = int(bins)
        centers = torch.linspace(self.vmin, self.vmax, self.bins)
        self.register_buffer("centers", centers)
        if lengthscale is None:
            step = (self.vmax - self.vmin) / max(self.bins - 1, 1)
            self.gamma = 1.0 / max(step, 1e-8)
        else:
            self.gamma = 1.0 / (float(lengthscale) ** 2)

    def forward(self, distance: Tensor) -> Tensor:
        if distance.dim() == 1:
            distance = distance.unsqueeze(-1)
        return torch.exp(-self.gamma * (distance - self.centers) ** 2)


class _MLPBN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return F.silu(self.bn(self.lin(x)))


class _EdgeGatedGraphConv(nn.Module):
    def __init__(self, input_features, output_features, residual=True):
        super().__init__()
        self.residual = bool(residual and input_features == output_features)
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(self, node_feats, edge_feats, edge_index):
        idx_i, idx_j = edge_index
        dim_size = node_feats.shape[0]
        e_src = self.src_gate(node_feats)[idx_i]
        e_dst = self.dst_gate(node_feats)[idx_j]
        y = e_src + e_dst + self.edge_gate(edge_feats)
        sigma = torch.sigmoid(y)
        bh = self.dst_update(node_feats)[idx_j]
        sum_sigma_h = scatter_add(bh * sigma, idx_i, dim=0, dim_size=dim_size)
        sum_sigma = scatter_add(sigma, idx_i, dim=0, dim_size=dim_size)
        h = sum_sigma_h / (sum_sigma + 1e-6)
        x = self.src_update(node_feats) + h
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(y))
        if self.residual:
            x = node_feats + x
            y = edge_feats + y
        return x, y


class _ALIGNNConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.node_update = _EdgeGatedGraphConv(hidden, hidden)
        self.edge_update = _EdgeGatedGraphConv(hidden, hidden)

    def forward(self, x, y, z, edge_index, line_edge_index):
        m, z = self.edge_update(y, z, line_edge_index)
        x, y = self.node_update(x, m, edge_index)
        return x, y, z


class ALIGNNMultiTask(nn.Module):
    """ALIGNN backbone with the shared HOA + isotherm output contract."""

    def __init__(
        self,
        node_input_features=1,
        embedding_features=64,
        triplet_input_features=40,
        hidden=256,
        out_size=1,
        centers=80,
        a_layers=4,
        g_layers=4,
        iso_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.atom_embedding = nn.Sequential(_MLPBN(node_input_features, hidden))
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=8.0, bins=centers),
            _MLPBN(centers, embedding_features),
            _MLPBN(embedding_features, hidden),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1.0, vmax=1.0, bins=triplet_input_features),
            _MLPBN(triplet_input_features, embedding_features),
            _MLPBN(embedding_features, hidden),
        )
        self.alignn_layers = nn.ModuleList([_ALIGNNConv(hidden) for _ in range(a_layers)])
        self.gcn_layers = nn.ModuleList([_EdgeGatedGraphConv(hidden, hidden) for _ in range(g_layers)])
        self.out_layer = nn.Linear(hidden, out_size)
        self.iso_layer = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.ELU(),
            nn.Dropout(iso_dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    @staticmethod
    def _line_graph(edge_index: Tensor) -> Tensor:
        """Create ordered bond pairs sharing the same central/source node.

        The radius graph may contain multiple periodic images of the same atom,
        so this implementation preserves edge multiplicity instead of coalescing
        the adjacency through a sparse matrix.
        """
        src = edge_index[0]
        pairs_u = []
        pairs_v = []
        for node in torch.unique(src):
            ids = torch.nonzero(src == node, as_tuple=False).flatten()
            m = ids.numel()
            if m <= 1:
                continue
            u = ids.repeat_interleave(m)
            v = ids.repeat(m)
            keep = u != v
            pairs_u.append(u[keep])
            pairs_v.append(v[keep])
        if not pairs_u:
            return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return torch.stack([torch.cat(pairs_u), torch.cat(pairs_v)], dim=0)

    @staticmethod
    def _bond_cosines(line_edge_index: Tensor, edge_vec: Tensor) -> Tensor:
        if line_edge_index.numel() == 0:
            return edge_vec.new_zeros((0, 1))
        u, v = line_edge_index
        r1 = edge_vec[u]
        r2 = edge_vec[v]
        denom = torch.linalg.norm(r1, dim=-1) * torch.linalg.norm(r2, dim=-1)
        cosine = (r1 * r2).sum(dim=-1) / denom.clamp_min(1e-12)
        return cosine.clamp(-1.0, 1.0).unsqueeze(-1)

    def forward(self, data, pres=None, integrate=False):
        if pres is None:
            raise ValueError("ALIGNNMultiTask requires pres for the shared isotherm head")
        if not hasattr(data, "edge_vec"):
            raise RuntimeError("data.edge_vec is missing; rebuild graphs with the patched data_utils.py")

        line_edge_index = self._line_graph(data.edge_index)
        if line_edge_index.numel() == 0:
            raise RuntimeError("ALIGNN line graph contains no triplets")
        angle = self._bond_cosines(line_edge_index, data.edge_vec)

        x = self.atom_embedding(data.x.float())
        y = self.edge_embedding(data.edge_attr.float())
        z = self.angle_embedding(angle.float())

        for layer in self.alignn_layers:
            x, y, z = layer(x, y, z, data.edge_index, line_edge_index)
        for layer in self.gcn_layers:
            x, y = layer(x, y, data.edge_index)

        graph_h = scatter_mean(x, data.batch, dim=0)
        hoa = self.out_layer(graph_h)

        p = pres.unsqueeze(0).repeat(graph_h.size(0), 1, 1)
        h_rep = graph_h.unsqueeze(1).repeat(1, p.shape[1], 1)
        hoa_rep = hoa.unsqueeze(1).repeat(1, p.shape[1], 1)
        q_prime = self.iso_layer(torch.cat([h_rep, p, hoa_rep], dim=-1)).squeeze(-1)
        return hoa, q_prime


class MatformerConv(MessagePassing):
    """PyG-style Matformer attention block used by the shared benchmark."""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=4,
        concat=True,
        dropout=0.0,
        edge_dim=None,
        bias=True,
        root_weight=True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        self.out_channels = int(out_channels)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)
        self.root_weight = bool(root_weight)

        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.lin_skip = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_concate = nn.Linear(heads * out_channels, out_channels)
        self.lin_msg_update = nn.Linear(out_channels * 3, out_channels * 3)
        self.msg_layer = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.LayerNorm(out_channels),
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.layer_norm = nn.LayerNorm(out_channels * 3)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor):
        H, C = self.heads, self.out_channels
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )
        out = out.reshape(-1, H * C)
        out = self.lin_concate(out)
        out = F.silu(self.bn(out))
        if self.root_weight:
            out = out + self.lin_skip(x)
        return out

    def message(
        self,
        query_i: Tensor,
        key_i: Tensor,
        key_j: Tensor,
        value_i: Tensor,
        value_j: Tensor,
        edge_attr: OptTensor,
    ) -> Tensor:
        H, C = self.heads, self.out_channels
        e = self.lin_edge(edge_attr).view(-1, H, C)
        q = torch.cat((query_i, query_i, query_i), dim=-1)
        k = torch.cat((key_i, key_j, e), dim=-1)
        alpha = (q * k) / math.sqrt(C * 3.0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = torch.cat((value_i, value_j, e), dim=-1)
        gate = torch.sigmoid(self.layer_norm(alpha))
        return self.msg_layer(self.lin_msg_update(msg) * gate)


class MatformerMultiTask(nn.Module):
    """Matformer backbone with the shared HOA + isotherm output contract."""

    def __init__(
        self,
        node_input_features=1,
        hidden=128,
        edge_features=128,
        num_layers=5,
        heads=4,
        dropout=0.0,
        out_size=1,
        iso_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.atom_embedding = nn.Linear(node_input_features, hidden)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=8.0, bins=edge_features),
            nn.Linear(edge_features, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )
        self.layers = nn.ModuleList([
            MatformerConv(
                in_channels=hidden,
                out_channels=hidden,
                heads=heads,
                edge_dim=hidden,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.out_layer = nn.Linear(hidden, out_size)
        self.iso_layer = nn.Sequential(
            nn.Linear(hidden + 2, hidden),
            nn.ELU(),
            nn.Dropout(iso_dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(),
        )

    def forward(self, data, pres=None, integrate=False):
        if pres is None:
            raise ValueError("MatformerMultiTask requires pres for the shared isotherm head")
        x = self.atom_embedding(data.x.float())
        e = self.edge_embedding(data.edge_attr.float())
        for layer in self.layers:
            x = layer(x, data.edge_index, e)
        graph_h = self.fc(scatter_mean(x, data.batch, dim=0))
        hoa = self.out_layer(graph_h)
        p = pres.unsqueeze(0).repeat(graph_h.size(0), 1, 1)
        h_rep = graph_h.unsqueeze(1).repeat(1, p.shape[1], 1)
        hoa_rep = hoa.unsqueeze(1).repeat(1, p.shape[1], 1)
        q_prime = self.iso_layer(torch.cat([h_rep, p, hoa_rep], dim=-1)).squeeze(-1)
        return hoa, q_prime
