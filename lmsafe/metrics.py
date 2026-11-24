"""
Evaluation metrics.

Everything here consumes a list of `WindowScores` -- one entry per test
window, each holding that window's per-edge anomaly scores alongside its
ground-truth labels and event-count weights -- in the same time order
`lmsafe.runner` produced them in. Keeping the window structure (instead
of flattening immediately into one global array) is what makes two
things possible that a flat array cannot support correctly:

1. **A per-window alert budget.** "Rec@B" in this project means: a SOC
   analyst can review at most `B` alerts *per window*, not `B` alerts
   total across the whole multi-week test stream (which would be an
   unrealistic, hindsight-only budget). `recall_at_budget` below ranks
   each window's edges independently and takes each window's own top-B.

2. **Correct, time-aware bootstrap.** Authentication events inside one
   window, and consecutive windows inside one attack episode, are
   strongly correlated -- they are not i.i.d. draws. `block_bootstrap_ci`
   resamples contiguous *blocks of windows* (a standard moving-block
   bootstrap for time series) rather than individual edges, so its
   confidence intervals are not artificially narrow. `episode_bootstrap_ci`
   is the complementary tool for per-episode summary statistics (such as
   detection delay) where the natural unit of resampling is "one
   episode", not "one window" or "one edge".

Score convention (shared with `lmsafe.detectors`): **higher score =
more anomalous**, so every function here ranks/thresholds on
`scores` directly, with no sign flip.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .data.windows import WindowGraph


@dataclass
class WindowScores:
    """One test window's scores, labels, and event-count weights,
    aligned by index to each other (and to the `WindowGraph.edges` they
    came from, though the `WindowGraph` itself is not kept here).

    Attributes
    ----------
    window_start : int
        Same clock as `Event.t` / `WindowGraph.window_start`.
    scores : np.ndarray
        Anomaly scores, higher = more anomalous.
    is_malicious : np.ndarray (bool)
        Ground truth. This is the *only* place in the whole package
        where it is legitimate to read `WindowGraph.edge_is_malicious`
        -- `lmsafe.runner` builds one `WindowScores` per window
        specifically to hand ground truth to `lmsafe.metrics` and
        nowhere else.
    weight : np.ndarray (int)
        Event count each row represents (`WindowGraph.edge_weight`) --
        used by the `weighted_*` metrics to recover event-level
        statistics from edge-deduplicated rows without having to
        re-materialize every individual event.
    """

    window_start: int
    scores: np.ndarray
    is_malicious: np.ndarray
    weight: np.ndarray

    @classmethod
    def from_window(cls, window: WindowGraph, scores: np.ndarray) -> "WindowScores":
        return cls(
            window_start=window.window_start,
            scores=np.asarray(scores, dtype=np.float64),
            is_malicious=np.asarray(window.edge_is_malicious, dtype=bool),
            weight=np.asarray(window.edge_weight, dtype=np.float64),
        )


def _concat(records: Sequence[WindowScores]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        return np.empty(0), np.empty(0, dtype=bool), np.empty(0)
    scores = np.concatenate([r.scores for r in records])
    labels = np.concatenate([r.is_malicious for r in records])
    weights = np.concatenate([r.weight for r in records])
    return scores, labels, weights


def weighted_auc(records: Sequence[WindowScores]) -> float:
    """Event-level ROC-AUC: every row's contribution is multiplied by
    its event count (`weight`), via `sklearn`'s native `sample_weight`
    support -- recovering the AUC you would get by scoring every
    original (non-deduplicated) event individually, without ever
    materializing that larger array.

    Returns `nan` if `records` contains only one class (no malicious
    edges at all, or -- pathologically on a tiny bootstrap block --
    no benign edges), since AUC is undefined in that case. Callers
    doing a bootstrap (see `block_bootstrap_ci`) should expect and
    filter `nan` draws rather than treat them as zero.
    """
    scores, labels, weights = _concat(records)
    if len(labels) == 0 or labels.all() or not labels.any():
        return float("nan")
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def weighted_ap(records: Sequence[WindowScores]) -> float:
    """Event-level average precision (area under the precision-recall
    curve), analogous to `weighted_auc`. `nan` if there are no
    malicious events at all in `records`."""
    scores, labels, weights = _concat(records)
    if len(labels) == 0 or not labels.any():
        return float("nan")
    return float(average_precision_score(labels, scores, sample_weight=weights))


def recall_at_budget(
    records: Sequence[WindowScores],
    budget_per_window: int,
    *,
    weighted: bool = True,
) -> float:
    """Recall at a fixed **per-window** alert budget ("Rec@B").

    For every window, the `budget_per_window` highest-scoring rows are
    "alerts" (ties broken by the stable order `numpy.argsort` returns,
    which for a deterministic detector is itself deterministic).
    Recall is accumulated across the whole stream as
    ``caught / total_malicious``, where both are summed in *event*
    units if `weighted=True` (using each row's `weight`) or in *row*
    (deduplicated-edge) units if `weighted=False`.

    A window with zero malicious rows contributes to neither the
    numerator nor the denominator (it can only ever generate false
    positives, which this metric does not score -- see `weighted_ap`/
    `weighted_auc` for precision-sensitive metrics). Returns `nan` if
    no window in `records` contains any malicious row.
    """
    caught = 0.0
    total = 0.0
    for r in records:
        if not r.is_malicious.any():
            continue
        n = len(r.scores)
        b = min(budget_per_window, n)
        top = np.argsort(-r.scores)[:b]
        alerted = np.zeros(n, dtype=bool)
        alerted[top] = True
        w = r.weight if weighted else np.ones(n)
        total += float(w[r.is_malicious].sum())
        caught += float(w[r.is_malicious & alerted].sum())
    return caught / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Episode clustering and detection delay
# ---------------------------------------------------------------------------


def cluster_episodes(
    records: Sequence[WindowScores], gap_windows: int = 1
) -> list[list[tuple[int, int]]]:
    """Group malicious rows into attack episodes by temporal proximity.

    Two malicious *windows* (not individual rows) belong to the same
    episode if they are separated by at most `gap_windows` windows with
    no malicious activity at all. This is the same gap-based clustering
    this project used to inspect the real LANL red-team file by hand
    (there, a 6-hour gap separated 19 episodes and a 24-hour gap
    separated 8) -- generalized here to operate in units of windows so
    it works at any `window_seconds`.

    Returns
    -------
    list[list[tuple[int, int]]]
        One list per episode, each containing `(record_index, edge_index)`
        pairs identifying every malicious row in that episode, in the
        order they occur. `record_index` indexes into `records`.
    """
    malicious_positions = sorted(i for i, r in enumerate(records) if r.is_malicious.any())
    if not malicious_positions:
        return []

    groups: list[list[int]] = []
    current = [malicious_positions[0]]
    for pos in malicious_positions[1:]:
        if pos - current[-1] > gap_windows + 1:
            groups.append(current)
            current = []
        current.append(pos)
    groups.append(current)

    episodes: list[list[tuple[int, int]]] = []
    for group in groups:
        edges = [
            (i, j)
            for i in group
            for j in np.flatnonzero(records[i].is_malicious)
        ]
        episodes.append(edges)
    return episodes


def detection_delay_per_episode(
    records: Sequence[WindowScores],
    budget_per_window: int,
    episodes: Sequence[Sequence[tuple[int, int]]],
) -> list[Optional[int]]:
    """For each episode, the number of windows between the episode's
    first malicious window and the first window in which *any* of that
    episode's malicious rows is caught in the per-window top-B alerts.

    Returns `None` for an episode never caught within `records` (the
    caller should report "fraction of episodes never caught at all" as
    a distinct statistic -- see module docstring -- rather than silently
    dropping these; `episode_bootstrap_ci` expects `None`s to already be
    filtered out, by design, so that an unbounded "delay" does not
    distort a mean).
    """
    delays: list[Optional[int]] = []
    for edges in episodes:
        if not edges:
            delays.append(None)
            continue
        first_window = min(i for i, _ in edges)
        by_window: dict[int, list[int]] = defaultdict(list)
        for i, j in edges:
            by_window[i].append(j)

        caught_at = None
        for i in sorted(by_window):
            r = records[i]
            n = len(r.scores)
            b = min(budget_per_window, n)
            top = set(np.argsort(-r.scores)[:b].tolist())
            if any(j in top for j in by_window[i]):
                caught_at = i
                break
        delays.append(None if caught_at is None else caught_at - first_window)
    return delays


def episode_hit(
    records: Sequence[WindowScores],
    budget_per_window: int,
    episodes: Sequence[Sequence[tuple[int, int]]],
) -> list[bool]:
    """Per-episode binary indicator: was this episode caught at all
    (any malicious row inside the per-window top-B, at any point)?
    A convenient, bootstrap-able companion to `detection_delay_per_episode`
    for episodes that are never caught at all (whose delay is `None`).
    """
    return [d is not None for d in detection_delay_per_episode(records, budget_per_window, episodes)]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    metric_fn: Callable[[Sequence[WindowScores]], float],
    records: Sequence[WindowScores],
    *,
    block_size: int = 24,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block-bootstrap confidence interval for a window-level
    metric (`weighted_auc`, `weighted_ap`, or a `recall_at_budget`
    partial application).

    Resamples `len(records) // block_size` contiguous blocks of
    `block_size` consecutive windows, with replacement, concatenates
    them into one bootstrap replicate, and recomputes `metric_fn` on
    it -- preserving within-block time correlation instead of treating
    windows (or worse, individual edges) as independent. `block_size`
    should be chosen comfortably larger than the attack episodes you
    care about (e.g. >= the longest episode duration in windows) so a
    block only rarely splits an episode in half.

    Draws where `metric_fn` returns `nan` (e.g. a block with no
    malicious rows at all, for AUC/AP) are dropped before computing
    percentiles; `n_valid` fewer than `n_boot` is reported implicitly
    by a wider-than-expected interval and should be checked by the
    caller if `n_boot` is small.

    Returns
    -------
    (point_estimate, lo, hi)
        `point_estimate` is `metric_fn(records)` on the *original*
        (non-resampled) data; `lo`/`hi` are the `ci` percentile bounds
        of the bootstrap distribution.
    """
    point = metric_fn(records)
    n = len(records)
    if n == 0:
        return point, float("nan"), float("nan")
    block_size = min(block_size, n)
    n_blocks = max(1, n // block_size)
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0

    boot_vals = []
    max_start = max(n - block_size, 0)
    for _ in range(n_boot):
        resampled: list[WindowScores] = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, max_start + 1))
            resampled.extend(records[start : start + block_size])
        val = metric_fn(resampled)
        if not np.isnan(val):
            boot_vals.append(val)

    if not boot_vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boot_vals, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lo), float(hi)


def episode_bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over a list of per-episode scalar values
    (e.g. `detection_delay_per_episode` output with `None`s already
    filtered out, or `episode_hit` cast to 0/1).

    This is the right tool specifically when the natural "unit of
    independence" is one episode -- as opposed to `block_bootstrap_ci`,
    which resamples contiguous time blocks and is for metrics (AUC, AP,
    Rec@B) computed over the full mix of benign and malicious rows.
    Mixing these up -- e.g. bootstrapping Rec@B by resampling episodes
    alone, ignoring the benign majority of the stream -- would
    understate the true uncertainty and is exactly the mistake this
    project's evaluation protocol is designed to avoid (see
    `lmsafe.config` module docstring and README "Why a fair split
    matters").
    """
    vals = np.asarray(list(values), dtype=np.float64)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(vals.mean())
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    boot_means = [
        float(vals[rng.integers(0, n, size=n)].mean()) for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return point, float(lo), float(hi)
