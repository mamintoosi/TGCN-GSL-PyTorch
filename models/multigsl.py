"""
Canonical T-GCN multi-lag graph-structure-learning models.

This module is the SINGLE home of the model implementations used by the
revised manuscript. The classes below are copied verbatim from
gsl_stage26/stage26_validation.py (the canonical Stage 26/29 training path),
so moving them here does not change any numeric behaviour.

Canonical naming (manuscript terminology)
-----------------------------------------
Manuscript name        canonical id     implementation class          legacy names
---------------------------------------------------------------------------------
T-GCN-NoSpatial        no_spatial       models.tgcn.TGCN (identity)   NoGraph, nograph,
                                                                      standard, NoGraph_h64,
                                                                      NoGraph_h74
Physical               physical         models.tgcn.TGCN (road net)   Physical
T-GCN-MultiGSL         multi_gsl        MultiGraphTGCNFixed           MultiGraphTGCN_fixed,
                                                                      multi_graph_fixed,
                                                                      MultiGraphTGCN,
                                                                      MultiGraphTGCN_thr0.1
T-GCN-MultiGSL-Mix     multi_gsl_mix    GatedMultiGraphTGCN           GatedMultiGraphTGCN,
                                                                      gated_multi,
                                                                      GatedMulti_thr0.1,
                                                                      GatedMulti
T-GCN-MultiGSL-Weighted multi_gsl_weighted WeightedMultiGraphTGCN     WeightedMultiGraphTGCN,
                       (supplementary)                                weighted_multi,
                                                                      WeightedMulti_thr0.1

The full mapping (including historical stage scripts and result keys) lives in
doc/METHOD_NAMING_MAP.md. Use normalize_method() to translate any legacy name
to its canonical id; never rename historical result artifacts.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tgcn import TGCN
from utils.graph_conv import calculate_laplacian_with_self_loop


# ============================================================
# T-GCN-MultiGSL-Mix (proposed method)
# ============================================================
class GatedMultiGraphTGCN(nn.Module):
    """Per-node, per-timestep learned mixing over lag-specific graphs.

    Manuscript name: T-GCN-MultiGSL-Mix.
    At each input timestep a gate network computes per-node softmax weights
    over the K lag-graph Laplacians and mixes them before the GRU update.
    """

    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(adj)) for adj in adj_list]
        self.register_buffer("lap_stack", torch.stack(laps))
        self.gate_net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._n_graphs),
        )
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        for t in range(T):
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)
            gate_input = torch.cat([x, hh], dim=2)
            gate_logits = self.gate_net(gate_input)
            gate_w = F.softmax(gate_logits, dim=-1)
            adj_weighted = torch.einsum('bnk,kij->bnj', gate_w, self.lap_stack)
            gh = torch.cat([x, hh], dim=2)
            ag = torch.bmm(adj_weighted, gh)
            z = torch.sigmoid(self.W_z(ag))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


# ============================================================
# T-GCN-MultiGSL (fixed lag-graph assignment)
# ============================================================
class MultiGraphTGCNFixed(nn.Module):
    """Fixed assignment of lag-specific graphs to input timesteps.

    Manuscript name: T-GCN-MultiGSL.

    Mapping (corrected alignment, same as Stage 26):
      input step t (0 = most recent) -> lag graph for temporal gap (T-1-t)
      gap 1 -> lag_1 graph (index 0), gap 2 -> lag_2 (index 1), gap 3 -> lag_3
      (index 2), larger gaps cycle: index = gap % n_graphs.
    """

    def __init__(self, adj_list, hidden_dim=64, seq_len=12, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        self._seq_len = seq_len
        laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(adj)) for adj in adj_list]
        for i, lap in enumerate(laps):
            self.register_buffer(f"lap_{i}", lap)
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def _graph_conv(self, lap, x):
        B, N, D = x.shape
        x_flat = x.permute(1, 2, 0).reshape(N, D * B)
        out = lap @ x_flat
        return out.reshape(N, D, B).permute(2, 0, 1)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        for t in range(T):
            temporal_gap = (T - 1) - t
            graph_idx = temporal_gap % self._n_graphs
            lap = getattr(self, f"lap_{graph_idx}")
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)
            gh = self._graph_conv(lap, torch.cat([x, hh], dim=2))
            z = torch.sigmoid(self.W_z(gh))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}


# ============================================================
# T-GCN-MultiGSL-Weighted (supplementary ablation)
# ============================================================
class WeightedMultiGraphTGCN(nn.Module):
    """Learnable global scalar weights over the lag-specific graphs.

    Manuscript name: T-GCN-MultiGSL-Weighted (supplementary ablation only;
    appears in Stage 26 evaluation artifacts, not in the main results tables).
    A = sum_k softmax(w)_k * A_k with one learnable scalar per lag graph.
    """

    def __init__(self, adj_list, hidden_dim=64, **kwargs):
        super().__init__()
        self._input_dim = adj_list[0].shape[0]
        self._hidden_dim = hidden_dim
        self._n_graphs = len(adj_list)
        laps = [calculate_laplacian_with_self_loop(torch.FloatTensor(adj)) for adj in adj_list]
        self.register_buffer("lap_stack", torch.stack(laps))
        self.log_weights = nn.Parameter(torch.zeros(self._n_graphs))
        self.W_z = nn.Linear(1 + hidden_dim, hidden_dim * 2)
        self.W_n = nn.Linear(1 + hidden_dim, hidden_dim)

    def forward(self, inputs):
        B, T, N = inputs.shape
        h = torch.zeros(B, N * self._hidden_dim, device=inputs.device, dtype=inputs.dtype)
        w = torch.softmax(self.log_weights, dim=0)
        lap = torch.einsum('k,kij->ij', w, self.lap_stack)
        for t in range(T):
            x = inputs[:, t, :].reshape(B, N, 1)
            hh = h.reshape(B, N, self._hidden_dim)
            gh = torch.cat([x, hh], dim=2)
            D_B = (1 + self._hidden_dim) * B
            gh_flat = gh.permute(1, 2, 0).reshape(N, D_B)
            ag = lap @ gh_flat
            ag = ag.reshape(N, 1 + self._hidden_dim, B).permute(2, 0, 1)
            z = torch.sigmoid(self.W_z(ag))
            r, u = torch.chunk(z, chunks=2, dim=2)
            c = torch.tanh(self.W_n(torch.cat([x, r * hh], dim=2)))
            h = u * hh + (1 - u) * c
        return h.reshape(B, N, self._hidden_dim)

    @property
    def hyperparameters(self):
        return {"hidden_dim": self._hidden_dim}

    def get_graph_weights(self):
        return torch.softmax(self.log_weights, dim=0).detach().cpu().numpy()


# ============================================================
# Method registry
# ============================================================
# adjacency: how the method consumes its graph argument in build_model()
#   "identity"  - fixed identity adjacency (no graph)
#   "single"    - one (N, N) adjacency matrix
#   "lag_list"  - list of K lag-specific (N, N) adjacency matrices
METHOD_REGISTRY = {
    # ------------------------------------------------------------------
    # T-GCN family (recurrent backbone: TGCN)
    # ------------------------------------------------------------------
    "no_spatial": {
        "name": "T-GCN-NoSpatial",
        "adjacency": "identity",
        "backbone": "tgcn",
        "legacy": ["NoGraph", "nograph", "standard", "NoGraph_h64", "NoGraph_h74",
                   "NoSpatial", "T-GCN-NoSpatial"],
    },
    "physical": {
        "name": "T-GCN",
        "adjacency": "single",
        "backbone": "tgcn",
        "legacy": ["Physical", "phys", "TGCN"],
    },
    "gsl": {
        "name": "T-GCN-GSL",
        "adjacency": "single",
        "backbone": "tgcn",
        "dagma_type": "contemporaneous",
        "legacy": ["gsl", "GSL", "T-GCN-GSL"],
    },
    "cgsl": {
        "name": "T-GCN-cGSL",
        "adjacency": "single",
        "backbone": "tgcn",
        "dagma_type": "contemporaneous",
        "construction": "A + A.T, threshold > 0",
        "legacy": ["cgsl", "cGSL", "T-GCN-cGSL"],
    },
    "multi_gsl": {
        "name": "T-GCN-MultiGSL",
        "adjacency": "lag_list",
        "backbone": "tgcn",
        "cls": MultiGraphTGCNFixed,
        "legacy": ["MultiGraphTGCN_fixed", "multi_graph_fixed", "MultiGraphTGCN",
                   "MultiGraphTGCN_thr0.1", "MultiGraph", "T-GCN-MultiGSL"],
    },
    "multi_gsl_mix": {
        "name": "T-GCN-MultiGSL-Mix",
        "adjacency": "lag_list",
        "backbone": "tgcn",
        "cls": GatedMultiGraphTGCN,
        "legacy": ["GatedMultiGraphTGCN", "gated_multi", "GatedMulti_thr0.1",
                   "GatedMulti", "T-GCN-MultiGSL-Mix"],
    },
    "multi_gsl_weighted": {
        "name": "T-GCN-MultiGSL-Weighted",
        "adjacency": "lag_list",
        "backbone": "tgcn",
        "cls": WeightedMultiGraphTGCN,
        "legacy": ["WeightedMultiGraphTGCN", "weighted_multi", "WeightedMulti_thr0.1",
                   "WeightedMulti"],
    },
    # ------------------------------------------------------------------
    # GCN family (non-recurrent backbone: GCN)
    # NOTE: GCN-MultiGSL / GCN-MultiGSL-Weighted / GCN-MultiGSL-Mix are
    # NOT included because the GCN architecture processes the entire input
    # window in a single graph-convolution step (no per-timestep recurrence).
    # Lag-specific graph assignment and per-timestep gating are therefore
    # architecturally meaningless for the GCN backbone.  See doc/STAGE40.
    # ------------------------------------------------------------------
    "gcn_physical": {
        "name": "GCN",
        "adjacency": "single",
        "backbone": "gcn",
        "legacy": ["gcn_physical", "GCN"],
    },
    "gcn_no_spatial": {
        "name": "GCN-NoSpatial",
        "adjacency": "identity",
        "backbone": "gcn",
        "legacy": ["gcn_no_spatial", "GCN-NoSpatial"],
    },
    "gcn_gsl": {
        "name": "GCN-GSL",
        "adjacency": "single",
        "backbone": "gcn",
        "dagma_type": "contemporaneous",
        "legacy": ["gcn_gsl", "GCN-GSL"],
    },
    "gcn_cgsl": {
        "name": "GCN-cGSL",
        "adjacency": "single",
        "backbone": "gcn",
        "dagma_type": "contemporaneous",
        "construction": "A + A.T, threshold > 0",
        "legacy": ["gcn_cgsl", "GCN-cGSL"],
    },
    # ------------------------------------------------------------------
    # GCN-MultiGSL: union of lag-specific graphs, single static graph to GCN
    # ------------------------------------------------------------------
    "gcn_multigsl": {
        "name": "GCN-MultiGSL",
        "adjacency": "single",
        "backbone": "gcn",
        "dagma_type": "multilag_union",
        "construction": "A_union = 1(max_l A_l > 0), threshold > 0.1 per lag then union",
        "legacy": ["gcn_multigsl", "GCN-MultiGSL"],
    },
}

# Flat legacy-name -> canonical-id lookup (built once at import)
_LEGACY_TO_CANONICAL = {}
for _cid, _meta in METHOD_REGISTRY.items():
    _LEGACY_TO_CANONICAL[_cid] = _cid
    for _lg in _meta["legacy"]:
        _LEGACY_TO_CANONICAL[_lg] = _cid


def normalize_method(name):
    """Translate any canonical id or legacy name to the canonical method id.

    Returns None for unknown names. Historical result artifacts keep their
    original keys; use this helper when reading them.
    """
    if name is None:
        return None
    return _LEGACY_TO_CANONICAL.get(str(name))


def canonical_name(method_id):
    """Canonical manuscript display name for a canonical method id."""
    return METHOD_REGISTRY[method_id]["name"]


def build_model(method_id, adj=None, adj_list=None, hidden_dim=64, seq_len=12):
    """Instantiate a canonical method.

    Parameters
    ----------
    method_id : canonical id (see METHOD_REGISTRY); legacy names are accepted
        via normalize_method().
    adj : (N, N) array  -- required for 'single' methods.
    adj_list : list of (N, N) arrays -- required for 'lag_list' methods.
        For 'identity' both may be omitted (identity adjacency is used).
    seq_len : input sequence length (required by GCN backbone).
    """
    method_id = normalize_method(method_id)
    if method_id is None:
        raise ValueError(f"Unknown method id: {method_id!r}")
    meta = METHOD_REGISTRY[method_id]
    kind = meta["adjacency"]
    backbone = meta.get("backbone", "tgcn")

    if kind == "identity":
        n = (adj_list[0].shape[0] if adj_list is not None
             else (adj.shape[0] if adj is not None else None))
        if n is None:
            raise ValueError("identity method requires adj or adj_list to infer N")
        if backbone == "gcn":
            from models.gcn import GCN as GCNClass
            return GCNClass(adj=np.eye(n, dtype=np.float32),
                            seq_len=seq_len, hidden_dim=hidden_dim)
        return TGCN(adj=np.eye(n, dtype=np.float32), hidden_dim=hidden_dim)
    if kind == "single":
        if adj is None:
            raise ValueError(f"{method_id} requires a single adjacency matrix")
        if backbone == "gcn":
            from models.gcn import GCN as GCNClass
            return GCNClass(adj=adj, seq_len=seq_len, hidden_dim=hidden_dim)
        return TGCN(adj=adj, hidden_dim=hidden_dim)
    # lag_list (T-GCN family only)
    if not adj_list:
        raise ValueError(f"{method_id} requires a list of lag-specific adjacencies")
    cls = meta["cls"]
    return cls(adj_list=adj_list, hidden_dim=hidden_dim)


# ============================================================
# Graph constructors shared by control experiments
# ============================================================
def binary_graph(W, threshold):
    """Threshold |W| into a binary adjacency and remove self-loops."""
    adj = (np.abs(W) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)
    return adj


def correlation_topk_graph(train_norm, k):
    """Top-k |Pearson| directed edges from TRAINING data only (no self-loops).

    Used by the Stage 32 sparse controls (CorrTop30) to test whether the
    multi-lag gains are explained by sparsity alone. Must only ever see
    training data.
    """
    N = train_norm.shape[1]
    C = np.corrcoef(train_norm.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    flat = np.abs(C).ravel()
    flat[np.arange(N) * N + np.arange(N)] = 0.0  # kill self-pairs
    idx = np.argsort(flat)[::-1][:k]
    adj = np.zeros((N, N), dtype=np.float32)
    adj.flat[idx] = 1.0
    return adj


def random_edge_graph(N, k, seed):
    """k random off-diagonal directed edges (deterministic per seed)."""
    rng = np.random.RandomState(seed)
    adj = np.zeros((N, N), dtype=np.float32)
    chosen = set()
    while len(chosen) < k:
        i, j = rng.randint(N), rng.randint(N)
        if i == j:
            continue
        chosen.add((i, j))
    for (i, j) in chosen:
        adj[i, j] = 1.0
    return adj
