"""
Lightweight, inductive link-prediction detector (torch, CPU-friendly).

Why not per-node ID embeddings
--------------------------------
CyberGFM-style models tokenize random walks over node *identities*: every
host gets its own learned embedding row, looked up by ID. That approach
cannot score an edge touching a host it never saw during training --
exactly the failure mode flagged in this project's literature review
(CyberGFM is "non-inductive"). An enterprise network's host population
drifts (RQ1's whole premise), so a detector that cannot even produce a
score for a brand-new host is unusable for this paper's purposes.

This detector instead represents each host by a small vector of *running,
incrementally-updated statistics* (out/in-degree, partner diversity, age,
recency), computed the same way whether the host was seen once or ten
thousand times. A host nobody has ever seen gets the all-zero "cold
start" feature vector, which is itself a meaningful, learnable signal
("unfamiliar host") rather than an undefined embedding lookup. This is a
small, defensible model appropriate for a CPU-only sandbox and for the
paper's "lightweight baseline GFM-style detector" role; swapping in a
heavier pretrained model later only requires implementing the same
`Detector` interface.

Training signal
-----------------
For every window handed to `fit`, every observed edge is a positive
example (label 1, "this edge is part of normal traffic"); a configurable
number of random host-pairs drawn from that window's node set are
negative examples (label 0). This is standard link-prediction negative
sampling -- note it is an *approximation*: a small fraction of "negative"
pairs will, by chance, coincide with a real (positive) edge, mislabeling
it. With a modest negative/positive ratio and many windows, this does not
meaningfully bias results, but it is a real property of the method and
is documented here rather than left implicit.

Scoring convention
--------------------
The model outputs a logit where higher = "looks like normal traffic"
(matching the positive-label training convention). `score()` negates it,
so -- matching every other detector in this project -- **higher returned
score = more anomalous**.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..data.windows import WindowGraph
from .base import Detector

HOST_FEATURE_DIM = 6
EDGE_FEATURE_DIM = 2 * HOST_FEATURE_DIM + 1


class _LinkPredictorMLP(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EDGE_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _HostStats:
    """Running per-host statistics, updated one window at a time."""

    __slots__ = ("out_count", "in_count", "out_partners", "in_partners", "first_seen", "last_seen")

    def __init__(self):
        self.out_count: dict[str, int] = {}
        self.in_count: dict[str, int] = {}
        self.out_partners: dict[str, set] = {}
        self.in_partners: dict[str, set] = {}
        self.first_seen: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}

    def feature(self, host: str, window_idx: int) -> np.ndarray:
        if host not in self.first_seen:
            return np.zeros(HOST_FEATURE_DIM, dtype=np.float32)
        age = window_idx - self.first_seen[host]
        recency = window_idx - self.last_seen[host]
        return np.array(
            [
                np.log1p(self.out_count.get(host, 0)),
                np.log1p(self.in_count.get(host, 0)),
                np.log1p(len(self.out_partners.get(host, ()))),
                np.log1p(len(self.in_partners.get(host, ()))),
                np.log1p(max(age, 0)),
                np.log1p(max(recency, 0)),
            ],
            dtype=np.float32,
        )

    def update(self, window: WindowGraph, window_idx: int) -> None:
        for (a, b), weight in zip(window.edges, window.edge_weight):
            src, dst = window.nodes[a], window.nodes[b]
            self.out_count[src] = self.out_count.get(src, 0) + weight
            self.in_count[dst] = self.in_count.get(dst, 0) + weight
            self.out_partners.setdefault(src, set()).add(dst)
            self.in_partners.setdefault(dst, set()).add(src)
            for h in (src, dst):
                self.first_seen.setdefault(h, window_idx)
                self.last_seen[h] = window_idx


class FeatureMLPDetector(Detector):
    """Small torch MLP link-predictor over running per-host features.

    Parameters
    ----------
    hidden_dim : int
        MLP hidden width. 16 is deliberately small -- this is meant to
        run in seconds per window on a 2-CPU sandbox, not to be a
        state-of-the-art architecture.
    n_negatives_per_positive : int
        Negative-sampling ratio used during `fit`.
    epochs_per_fit : int
        Number of full passes over the positive+negative examples built
        from the windows passed to one `fit` call.
    lr : float
        Adam learning rate.
    edge_count_cap : float
        Edge-level running counts are clipped to this value before
        `log1p` purely to keep the feature numerically bounded for
        extremely hot edges (e.g. a domain controller hit every
        window); does not affect ranking for any realistic count.
    seed : int
        Seeds both the torch model's initial weights and the numpy RNG
        used for negative sampling, for reproducible experiments.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        n_negatives_per_positive: int = 2,
        epochs_per_fit: int = 3,
        lr: float = 1e-2,
        edge_count_cap: float = 1e6,
        seed: int = 0,
    ):
        self.hidden_dim = hidden_dim
        self.n_negatives_per_positive = n_negatives_per_positive
        self.epochs_per_fit = epochs_per_fit
        self.lr = lr
        self.edge_count_cap = edge_count_cap
        self.seed = seed

        torch.manual_seed(seed)
        self._model = _LinkPredictorMLP(hidden_dim)
        self._opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._rng = np.random.default_rng(seed)

        self._stats = _HostStats()
        self._edge_counts: dict[tuple[str, str], int] = {}
        self._window_idx = 0

    # -- internal helpers ---------------------------------------------

    def _edge_features(self, window: WindowGraph, a: int, b: int) -> np.ndarray:
        src, dst = window.nodes[a], window.nodes[b]
        f_src = self._stats.feature(src, self._window_idx)
        f_dst = self._stats.feature(dst, self._window_idx)
        c = min(self._edge_counts.get((src, dst), 0), self.edge_count_cap)
        return np.concatenate([f_src, f_dst, [np.log1p(c)]]).astype(np.float32)

    def _build_training_batch(self, window: WindowGraph) -> tuple[torch.Tensor, torch.Tensor]:
        if not window.nodes or not window.edges:
            return torch.empty(0, EDGE_FEATURE_DIM), torch.empty(0)

        pos_edge_set = set(window.edges)
        feats: list[np.ndarray] = []
        labels: list[float] = []
        n_nodes = len(window.nodes)

        for (a, b) in window.edges:
            feats.append(self._edge_features(window, a, b))
            labels.append(1.0)
            for _ in range(self.n_negatives_per_positive):
                na = int(self._rng.integers(0, n_nodes))
                nb = int(self._rng.integers(0, n_nodes))
                # Approximation, documented in the module docstring: we
                # do not retry on a collision with a real positive edge
                # beyond this one cheap reroll, to keep sampling O(1)
                # per negative even on dense windows.
                if (na, nb) in pos_edge_set:
                    nb = (nb + 1) % n_nodes
                feats.append(self._edge_features(window, na, nb))
                labels.append(0.0)

        x = torch.from_numpy(np.stack(feats))
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y

    # -- Detector interface --------------------------------------------

    def fit(self, windows: Sequence[WindowGraph], *, warm_start: bool = True) -> None:
        if not warm_start:
            torch.manual_seed(self.seed)
            self._model = _LinkPredictorMLP(self.hidden_dim)
            self._opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
            self._stats = _HostStats()
            self._edge_counts = {}
            self._window_idx = 0

        for window in windows:
            if window.edges:
                x, y = self._build_training_batch(window)
                if x.shape[0] > 0:
                    self._model.train()
                    for _ in range(self.epochs_per_fit):
                        self._opt.zero_grad()
                        logits = self._model(x)
                        loss = self._loss_fn(logits, y)
                        loss.backward()
                        self._opt.step()

            for (a, b), weight in zip(window.edges, window.edge_weight):
                key = (window.nodes[a], window.nodes[b])
                self._edge_counts[key] = self._edge_counts.get(key, 0) + weight
            self._stats.update(window, self._window_idx)
            self._window_idx += 1

    def score(self, window: WindowGraph) -> np.ndarray:
        if not window.edges:
            return np.zeros(0, dtype=np.float64)
        feats = np.stack(
            [self._edge_features(window, a, b) for (a, b) in window.edges]
        )
        x = torch.from_numpy(feats)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x).numpy()
        return (-logits).astype(np.float64)

    def clone_fresh(self) -> "FeatureMLPDetector":
        return FeatureMLPDetector(
            hidden_dim=self.hidden_dim,
            n_negatives_per_positive=self.n_negatives_per_positive,
            epochs_per_fit=self.epochs_per_fit,
            lr=self.lr,
            edge_count_cap=self.edge_count_cap,
            seed=self.seed,
        )

    def state_summary(self) -> dict:
        return {
            "n_windows_fit": self._window_idx,
            "n_distinct_edges": len(self._edge_counts),
            "n_hosts_seen": len(self._stats.first_seen),
        }
