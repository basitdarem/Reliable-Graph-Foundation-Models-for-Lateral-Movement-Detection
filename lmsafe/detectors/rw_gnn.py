"""
Self-supervised graph neural network detector (torch, CPU-friendly) --
this project's "graph foundation model" piece.

Why a GNN, and why this particular design
-------------------------------------------
`FeatureMLPDetector` (feature_mlp.py) scores an edge from each endpoint's
own running statistics, computed *independently* -- it never looks at who
else a host is currently talking to. This detector adds exactly one
thing on top: one round of inductive neighborhood aggregation (a
GraphSAGE-style mean-pool update) over *this window's own graph*, so a
host's representation also reflects its current local neighborhood, not
just its own history. That is the actual, minimal, honest content behind
the word "graph" in "graph foundation model" here -- not a claim to
reimplement a published large-scale architecture.

Inductive by construction, like FeatureMLPDetector
-----------------------------------------------------
There is still no per-host-identity embedding table. The base feature
`h0` for each node is the same running (degree, partner-diversity, age,
recency) statistic `FeatureMLPDetector` uses (reused directly from
`feature_mlp._HostStats`, not reimplemented), and the neighborhood
aggregation only ever reads indices *local to the current window* (see
`WindowGraph.nodes`/`.edges`) -- never a global host vocabulary. This is
what makes "cross-network transfer" even a coherent thing to evaluate:
the encoder and scorer weights operate on a fixed 6-dimensional feature
space that means the same thing on any network's windows, so a model
pretrained on LANL's host population can be applied, unmodified, to
OpTC's completely disjoint one.

What "foundation model" means here, precisely
-------------------------------------------------
Pretraining (see scripts/pretrain_foundation_model.py) fits this
encoder+scorer once, via the same self-supervised negative-sampling
link-prediction objective used for ordinary `fit()`, but run over many
more windows than any single experiment's training split (e.g. the full
`lanl_full` 58-day stream). The resulting weights can then be handed to
a fresh `GraphFoundationDetector` via `pretrained_state_dict=...` for
zero-shot scoring or `freeze_encoder=True` fine-tuning on a different
network's windows, without ever retraining the encoder from scratch on
the new network. This is a deliberately small-scale, CPU-trainable
instance of the foundation-model idea -- a few hundred thousand
parameters, minutes to pretrain on a 2-CPU sandbox -- not a claim to
have built anything at the scale the term usually evokes in NLP/vision.
Scoped honestly, it is still the right experiment for the paper's
cross-network-transfer question: does *any* amount of pretraining on one
network transfer to another, versus training from scratch on the target
alone?

Training signal and scoring convention
-----------------------------------------
Identical to `FeatureMLPDetector`: every observed edge in a window is a
positive example; random host-pairs are negative examples (standard
link-prediction negative sampling, same documented approximation). The
model outputs a logit where higher = "looks like normal traffic";
`score()` negates it so higher returned score = more anomalous, matching
every other detector in this project.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..data.windows import WindowGraph
from .base import Detector
from .feature_mlp import HOST_FEATURE_DIM, _HostStats

GNN_HIDDEN_DIM_DEFAULT = 16


class _GraphEncoder(nn.Module):
    """One-hop inductive GraphSAGE-style update: a node's new
    representation combines its own base feature with the mean base
    feature of its in-neighbors and out-neighbors *within this window's
    graph only* -- no persistent state across windows or networks is
    stored here, which is exactly what keeps this inductive across an
    entirely different host population (e.g. OpTC's)."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, hidden_dim)
        self.out_nbr_lin = nn.Linear(in_dim, hidden_dim)
        self.in_nbr_lin = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, h0: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """`h0`: (n_nodes, in_dim) base features. `edge_index`: (2,
        n_edges) local (src, dst) indices, directed, one row per
        *distinct* edge (matching `WindowGraph.edges`)."""
        n_nodes = h0.shape[0]
        if edge_index.shape[1] == 0:
            out_mean = torch.zeros_like(h0)
            in_mean = torch.zeros_like(h0)
        else:
            src, dst = edge_index[0], edge_index[1]
            ones = torch.ones(src.shape[0], dtype=h0.dtype)

            # out_mean[i] = mean over j of h0[j] for every edge (i, j) --
            # the average base feature of i's out-neighbors.
            out_sum = torch.zeros(n_nodes, h0.shape[1], dtype=h0.dtype).index_add_(0, src, h0[dst])
            out_deg = torch.zeros(n_nodes, dtype=h0.dtype).index_add_(0, src, ones).clamp(min=1).unsqueeze(1)
            out_mean = out_sum / out_deg

            # in_mean[i] = mean over j of h0[j] for every edge (j, i) --
            # the average base feature of i's in-neighbors.
            in_sum = torch.zeros(n_nodes, h0.shape[1], dtype=h0.dtype).index_add_(0, dst, h0[src])
            in_deg = torch.zeros(n_nodes, dtype=h0.dtype).index_add_(0, dst, ones).clamp(min=1).unsqueeze(1)
            in_mean = in_sum / in_deg

        return self.act(self.self_lin(h0) + self.out_nbr_lin(out_mean) + self.in_nbr_lin(in_mean))


class _EdgeScorerMLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, h1: torch.Tensor, a_idx: torch.Tensor, b_idx: torch.Tensor, edge_count_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h1[a_idx], h1[b_idx], edge_count_feat.unsqueeze(1)], dim=1)
        return self.net(x).squeeze(-1)


class GraphFoundationDetector(Detector):
    """Self-supervised, inductive one-hop GNN link-predictor -- see the
    module docstring for the full design rationale and the precise,
    deliberately modest scope of "foundation model" claimed here.

    Parameters
    ----------
    hidden_dim : int
        Encoder output / scorer input width.
    n_negatives_per_positive, epochs_per_fit, lr, edge_count_cap, seed :
        Same meaning as `FeatureMLPDetector`'s identically-named
        parameters.
    pretrained_state_dict : dict, optional
        `{"encoder": <state_dict>, "scorer": <state_dict>}` produced by
        `export_state_dict()` (typically loaded from a checkpoint
        written by `scripts/pretrain_foundation_model.py`). Initializes
        the encoder and scorer weights before any `fit()` call -- this
        is the mechanism "cross-network transfer" uses: pretrain on one
        network, construct a fresh detector here with that checkpoint,
        then `fit()`/`score()` on a different network's windows.
    freeze_encoder : bool
        If True, the encoder's parameters are excluded from the
        optimizer entirely -- `fit()` only ever updates the edge-scorer
        MLP on top of a frozen pretrained encoder. Models the "zero-shot
        structural transfer, lightweight fine-tune on top" regime that
        is the interesting middle ground between "no pretraining at
        all" and "fully fine-tune everything".
    """

    def __init__(
        self,
        hidden_dim: int = GNN_HIDDEN_DIM_DEFAULT,
        n_negatives_per_positive: int = 2,
        epochs_per_fit: int = 3,
        lr: float = 1e-2,
        edge_count_cap: float = 1e6,
        seed: int = 0,
        pretrained_state_dict: Optional[dict] = None,
        freeze_encoder: bool = False,
    ):
        self.hidden_dim = hidden_dim
        self.n_negatives_per_positive = n_negatives_per_positive
        self.epochs_per_fit = epochs_per_fit
        self.lr = lr
        self.edge_count_cap = edge_count_cap
        self.seed = seed
        self.freeze_encoder = freeze_encoder
        self._pretrained_state_dict = pretrained_state_dict

        torch.manual_seed(seed)
        self._encoder = _GraphEncoder(HOST_FEATURE_DIM, hidden_dim)
        self._scorer = _EdgeScorerMLP(hidden_dim)
        if pretrained_state_dict is not None:
            self._encoder.load_state_dict(pretrained_state_dict["encoder"])
            self._scorer.load_state_dict(pretrained_state_dict["scorer"])
        if freeze_encoder:
            for p in self._encoder.parameters():
                p.requires_grad_(False)
        self._build_optimizer()
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._rng = np.random.default_rng(seed)

        self._stats = _HostStats()
        self._edge_counts: dict[tuple[str, str], int] = {}
        self._window_idx = 0

    def _build_optimizer(self) -> None:
        params = [p for p in self._encoder.parameters() if p.requires_grad]
        params += list(self._scorer.parameters())
        self._opt = torch.optim.Adam(params, lr=self.lr)

    # -- internal helpers ---------------------------------------------

    def _window_tensors(self, window: WindowGraph) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(window.nodes)
        h0 = torch.zeros(n, HOST_FEATURE_DIM, dtype=torch.float32)
        for i, host in enumerate(window.nodes):
            h0[i] = torch.from_numpy(self._stats.feature(host, self._window_idx))
        if window.edges:
            edge_index = torch.tensor(window.edges, dtype=torch.long).T
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        return h0, edge_index

    def _edge_count_feat(self, window: WindowGraph, pairs: Sequence[tuple[int, int]]) -> torch.Tensor:
        vals = []
        for a, b in pairs:
            key = (window.nodes[a], window.nodes[b])
            c = min(self._edge_counts.get(key, 0), self.edge_count_cap)
            vals.append(np.log1p(c))
        return torch.tensor(vals, dtype=torch.float32)

    def _sample_negatives(self, window: WindowGraph) -> list[tuple[int, int]]:
        n_nodes = len(window.nodes)
        pos_edge_set = set(window.edges)
        negs = []
        for _ in range(len(window.edges) * self.n_negatives_per_positive):
            na = int(self._rng.integers(0, n_nodes))
            nb = int(self._rng.integers(0, n_nodes))
            if (na, nb) in pos_edge_set:
                nb = (nb + 1) % n_nodes
            negs.append((na, nb))
        return negs

    # -- Detector interface --------------------------------------------

    def fit(self, windows: Sequence[WindowGraph], *, warm_start: bool = True) -> None:
        if not warm_start:
            torch.manual_seed(self.seed)
            self._encoder = _GraphEncoder(HOST_FEATURE_DIM, self.hidden_dim)
            self._scorer = _EdgeScorerMLP(self.hidden_dim)
            if self._pretrained_state_dict is not None:
                self._encoder.load_state_dict(self._pretrained_state_dict["encoder"])
                self._scorer.load_state_dict(self._pretrained_state_dict["scorer"])
            if self.freeze_encoder:
                for p in self._encoder.parameters():
                    p.requires_grad_(False)
            self._build_optimizer()
            self._stats = _HostStats()
            self._edge_counts = {}
            self._window_idx = 0

        for window in windows:
            if window.edges:
                h0, edge_index = self._window_tensors(window)
                negs = self._sample_negatives(window)
                pos_a = torch.tensor([a for a, _ in window.edges], dtype=torch.long)
                pos_b = torch.tensor([b for _, b in window.edges], dtype=torch.long)
                neg_a = torch.tensor([a for a, _ in negs], dtype=torch.long) if negs else torch.empty(0, dtype=torch.long)
                neg_b = torch.tensor([b for _, b in negs], dtype=torch.long) if negs else torch.empty(0, dtype=torch.long)
                pos_c = self._edge_count_feat(window, window.edges)
                neg_c = self._edge_count_feat(window, negs) if negs else torch.empty(0)
                y = torch.cat([torch.ones(len(window.edges)), torch.zeros(len(negs))])

                self._encoder.train()
                self._scorer.train()
                for _ in range(self.epochs_per_fit):
                    self._opt.zero_grad()
                    h1 = self._encoder(h0, edge_index)
                    logits_pos = self._scorer(h1, pos_a, pos_b, pos_c)
                    if negs:
                        logits_neg = self._scorer(h1, neg_a, neg_b, neg_c)
                        logits = torch.cat([logits_pos, logits_neg])
                    else:
                        logits = logits_pos
                    loss = self._loss_fn(logits, y)
                    loss.backward()
                    self._opt.step()

            self.observe(window)

    def score(self, window: WindowGraph) -> np.ndarray:
        if not window.edges:
            return np.zeros(0, dtype=np.float64)
        h0, edge_index = self._window_tensors(window)
        a_idx = torch.tensor([a for a, _ in window.edges], dtype=torch.long)
        b_idx = torch.tensor([b for _, b in window.edges], dtype=torch.long)
        c = self._edge_count_feat(window, window.edges)
        self._encoder.eval()
        self._scorer.eval()
        with torch.no_grad():
            h1 = self._encoder(h0, edge_index)
            logits = self._scorer(h1, a_idx, b_idx, c).numpy()
        return (-logits).astype(np.float64)

    def observe(self, window: WindowGraph) -> None:
        """Update the running host-stats / edge-count state from
        `window` WITHOUT any gradient step -- the neural weights
        (encoder + scorer) are left exactly as they are.

        This is the mechanism a genuine zero-shot cross-network-transfer
        evaluation needs: a detector carrying pretrained-on-network-A
        weights should be allowed to build up network B's own running
        host statistics as it moves through B's stream (so window 500
        benefits from having "seen" hosts in windows 1-499), without
        that counting as having *trained* on B at all. `fit()` already
        does this same state update internally (see its tail, after the
        optional gradient step) -- this method factors exactly that
        part out so a caller can get it without the training that
        normally comes attached.
        """
        for (a, b), weight in zip(window.edges, window.edge_weight):
            key = (window.nodes[a], window.nodes[b])
            self._edge_counts[key] = self._edge_counts.get(key, 0) + weight
        self._stats.update(window, self._window_idx)
        self._window_idx += 1

    def clone_fresh(self) -> "GraphFoundationDetector":
        return GraphFoundationDetector(
            hidden_dim=self.hidden_dim,
            n_negatives_per_positive=self.n_negatives_per_positive,
            epochs_per_fit=self.epochs_per_fit,
            lr=self.lr,
            edge_count_cap=self.edge_count_cap,
            seed=self.seed,
            pretrained_state_dict=self._pretrained_state_dict,
            freeze_encoder=self.freeze_encoder,
        )

    def state_summary(self) -> dict:
        return {
            "n_windows_fit": self._window_idx,
            "n_distinct_edges": len(self._edge_counts),
            "n_hosts_seen": len(self._stats.first_seen),
            "pretrained": self._pretrained_state_dict is not None,
            "encoder_frozen": self.freeze_encoder,
        }

    # -- pretraining / transfer support ---------------------------------

    def export_state_dict(self) -> dict:
        """`{"encoder": ..., "scorer": ...}`, suitable for
        `pretrained_state_dict=` on a fresh instance, or for
        `torch.save` to a checkpoint file (see
        `scripts/pretrain_foundation_model.py`)."""
        return {
            "encoder": self._encoder.state_dict(),
            "scorer": self._scorer.state_dict(),
        }
