"""
Non-parametric frequency baseline detector.

The model is deliberately the simplest thing that could possibly work:
track how many times each directed host-pair edge has been observed, and
score an edge as anomalous in inverse proportion to how often it has been
seen before, relative to total traffic volume. An edge never observed
before gets (close to) the maximum possible anomaly score.

This is not a strawman -- it is the reference point the paper needs. If
`FeatureMLPDetector` (or any future, heavier model) cannot clearly beat
this baseline on Rec@B, that is a real, reportable finding, not a bug.
Several papers in this literature compare only against learned models and
never report this kind of frequency baseline; this project does.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from ..data.windows import WindowGraph
from .base import Detector


class CountBaselineDetector(Detector):
    """Laplace-smoothed edge-frequency anomaly score.

    For an edge e with observed count c(e) out of `total` edge
    observations accumulated so far::

        p_hat(e)       = (c(e) + alpha) / (total + alpha)
        anomaly(e)     = -log(p_hat(e))

    `p_hat` is not a normalized probability distribution over all
    possible host pairs (that denominator would require knowing the full
    host population size, which drifts -- see lmsafe.synth); it is a
    monotonically-decreasing function of relative frequency, which is all
    that ranking-based metrics (AUC, AP, Rec@B) need. An unseen edge
    (c=0) gets anomaly = -log(alpha / (total + alpha)), the maximum
    possible value given the data seen so far; it grows (correctly)
    without bound as `total` grows and the edge is still never observed.

    Parameters
    ----------
    alpha : float
        Laplace smoothing constant. Smaller alpha makes "never seen
        before" edges score relatively higher versus "seen exactly
        once" edges; 0.5 (Jeffreys-style smoothing) is a reasonable,
        unremarkable default.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._counts: dict[tuple[str, str], int] = defaultdict(int)
        self._total: int = 0

    def fit(self, windows: Sequence[WindowGraph], *, warm_start: bool = True) -> None:
        if not warm_start:
            self._counts = defaultdict(int)
            self._total = 0
        for w in windows:
            for (a, b), weight in zip(w.edges, w.edge_weight):
                key = (w.nodes[a], w.nodes[b])
                self._counts[key] += weight
                self._total += weight

    def score(self, window: WindowGraph) -> np.ndarray:
        out = np.empty(len(window.edges), dtype=np.float64)
        # total+alpha could be 0 only if alpha==0 and nothing fit yet;
        # guard so an unfitted detector still returns finite scores
        # (rank-order is all that matters, but a NaN would poison every
        # downstream metric).
        denom = max(self._total + self.alpha, 1e-9)
        for i, (a, b) in enumerate(window.edges):
            key = (window.nodes[a], window.nodes[b])
            c = self._counts.get(key, 0)
            p_hat = (c + self.alpha) / denom
            out[i] = -math.log(max(p_hat, 1e-300))
        return out

    def clone_fresh(self) -> "CountBaselineDetector":
        return CountBaselineDetector(alpha=self.alpha)

    def state_summary(self) -> dict:
        return {"n_distinct_edges": len(self._counts), "total_events": self._total}
