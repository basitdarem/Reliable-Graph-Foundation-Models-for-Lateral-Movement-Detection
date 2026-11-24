"""
Update policies: the paper's central comparison axis.

Every policy answers the same question differently: *given the stream of
test windows arriving one at a time, when and on what data should the
detector's model of "normal" be refreshed?* `lmsafe.runner` drives all of
them through the exact same loop (score the window, then let the policy
decide what happens to the detector), so differences in RQ1/RQ2/RQ3
results come only from the policy logic below, never from a difference
in how they are driven.

    FrozenPolicy          Never updates after the initial fit. The RQ1
                           reference point for "drift cost": whatever
                           detection quality is lost over time under
                           this policy *is* the cost of standing still.
    PeriodicPolicy         Every `period` windows, retrain FROM SCRATCH
                           on exactly the most recent `lookback` windows
                           -- a clean slate each time, no memory beyond
                           the lookback. Accepts every window's data
                           unconditionally, including any attacker-
                           injected conditioning edges (RQ2's naive
                           baseline).
    SlidingReplayPolicy     Keeps a continuously-shifting trailing buffer
                            of the last `window_size` windows and
                            retrains from scratch on it every
                            `retrain_period` windows. Also accepts
                            everything unconditionally -- the second
                            RQ2 naive baseline, methodologically distinct
                            from Periodic (overlapping sliding window vs
                            disjoint batches).
    DriftTriggeredPolicy    Like Frozen, except a `lmsafe.drift` detector
                            watches a per-window summary statistic (mean
                            anomaly score by default) and triggers a
                            from-scratch retrain on recent history when
                            it fires. A third RQ2 naive baseline: it
                            reacts to *aggregate* shift, which is exactly
                            what a slow, individually-unremarkable
                            conditioning attack is designed to evade.
    GatedPolicy             This project's proposed defense. See its own
                            docstring below for the three mechanisms
                            (score-gated admission, delayed multi-host
                            corroboration, clean anchor memory) that
                            together answer RQ3.

All five share the `UpdatePolicy` interface so `lmsafe.runner` can treat
them interchangeably.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Callable, Optional, Sequence

import numpy as np

from .data.windows import WindowGraph, filter_edges
from .detectors.base import Detector
from .drift import PageHinkley, SimplifiedADWIN


class UpdatePolicy(ABC):
    """Common interface for all update policies.

    `lmsafe.runner`'s loop, once per test window, is exactly::

        scores = detector.score(window)
        record_metrics(window, scores)           # ground truth read ONLY here
        detector = policy.observe(window, scores, detector)

    and before that loop starts, exactly once::

        policy.fit_initial(detector, train_windows)

    No policy implementation is allowed to read `window.edge_is_malicious`
    -- see `lmsafe.data.schema`'s module docstring on the ground-truth
    boundary. `GatedPolicy`'s "corroboration" logic uses only the
    *detector's own scores* and *structural* multi-host agreement,
    never the ground-truth label; this is checked in
    tests/test_no_label_leak.py.
    """

    name: str = "update_policy"

    def fit_initial(self, detector: Detector, train_windows: Sequence[WindowGraph]) -> None:
        """Default initial fit: a full (from-scratch) fit on the benign
        training windows. Overridden only if a policy needs to also
        seed its own internal buffers (every concrete policy below
        does, and calls this base implementation first via `super()`).
        """
        detector.fit(train_windows, warm_start=False)

    @abstractmethod
    def observe(self, window: WindowGraph, scores: np.ndarray, detector: Detector) -> Detector:
        """Called once per test window, after `detector.score(window)`
        has already been computed and recorded by the caller. Returns
        the detector object to use for scoring the *next* window (which
        may be the same object, mutated in place -- all detectors here
        mutate in place and return `self` from `fit`-triggering code
        paths, but a policy is free to swap in a different detector
        instance, e.g. `detector.clone_fresh()`, if that is cleaner for
        its logic).
        """


# ---------------------------------------------------------------------------
# RQ1 reference point
# ---------------------------------------------------------------------------


class FrozenPolicy(UpdatePolicy):
    """Never updates. Whatever quality the initial 41-hour benign fit
    has is exactly the quality it keeps for the rest of the stream."""

    name = "frozen"

    def observe(self, window: WindowGraph, scores: np.ndarray, detector: Detector) -> Detector:
        return detector


# ---------------------------------------------------------------------------
# RQ2 naive baselines (accept every window unconditionally)
# ---------------------------------------------------------------------------


class PeriodicPolicy(UpdatePolicy):
    """Every `period` windows, discard the model and retrain from
    scratch on exactly the most recent `lookback` windows.

    Parameters
    ----------
    period : int
        Number of test windows between retrains.
    lookback : int, optional
        Number of most-recent windows (including the just-elapsed
        period) used for each retrain. Defaults to `period` itself,
        i.e. **disjoint, non-overlapping batches** -- each retrain sees
        only windows accumulated since the previous retrain and nothing
        older, which is the policy's defining property and the reason
        it is methodologically distinct from `SlidingReplayPolicy`
        below. Pass `lookback > period` deliberately for a variant that
        also keeps some older history (e.g. `lookback = 7 * period` for
        "retrain weekly on the last month"); at that point it starts to
        resemble `SlidingReplayPolicy` and the two are no longer a
        clean contrast, so RQ experiment configs should keep
        `lookback == period` (the default) unless that overlap is
        specifically what is being studied.
    """

    name = "periodic"

    def __init__(self, period: int = 24, lookback: Optional[int] = None):
        self.period = period
        self.lookback = lookback if lookback is not None else period
        self._buffer: deque[WindowGraph] = deque(maxlen=self.lookback)
        self._since_retrain = 0

    def fit_initial(self, detector: Detector, train_windows: Sequence[WindowGraph]) -> None:
        super().fit_initial(detector, train_windows)
        self._buffer.extend(train_windows)

    def observe(self, window: WindowGraph, scores: np.ndarray, detector: Detector) -> Detector:
        self._buffer.append(window)
        self._since_retrain += 1
        if self._since_retrain >= self.period:
            self._since_retrain = 0
            detector.fit(list(self._buffer), warm_start=False)
        return detector


class SlidingReplayPolicy(UpdatePolicy):
    """Keeps a continuously-shifting trailing window of the last
    `window_size` test windows (unconditionally, including any attacker
    traffic) and retrains from scratch on it every `retrain_period`
    windows.

    Methodologically distinct from `PeriodicPolicy`: the buffer here
    slides by one window at a time (every window ages out exactly
    `window_size` windows after it arrived), whereas Periodic's buffer
    is refilled in disjoint, non-overlapping batches of size `period`.
    """

    name = "sliding_replay"

    def __init__(self, window_size: int = 168, retrain_period: int = 24):
        self.window_size = window_size
        self.retrain_period = retrain_period
        self._buffer: deque[WindowGraph] = deque(maxlen=window_size)
        self._since_retrain = 0

    def fit_initial(self, detector: Detector, train_windows: Sequence[WindowGraph]) -> None:
        super().fit_initial(detector, train_windows)
        self._buffer.extend(train_windows)

    def observe(self, window: WindowGraph, scores: np.ndarray, detector: Detector) -> Detector:
        self._buffer.append(window)
        self._since_retrain += 1
        if self._since_retrain >= self.retrain_period:
            self._since_retrain = 0
            detector.fit(list(self._buffer), warm_start=False)
        return detector


class DriftTriggeredPolicy(UpdatePolicy):
    """Frozen, except an `lmsafe.drift` detector watches a per-window
    summary statistic of the detector's own scores and triggers a
    from-scratch retrain on recent history when it fires.

    This is the naive baseline RQ2's conditioning attack is specifically
    designed to defeat: a slow drip of individually-unremarkable edges
    is, by construction, meant to avoid any sharp *aggregate* shift in
    mean anomaly score, so a detector that only reacts to that aggregate
    shift should still end up absorbing the conditioning edges whenever
    it does eventually retrain (whether triggered by the real benign
    drift or by chance).

    Parameters
    ----------
    drift_detector : PageHinkley or SimplifiedADWIN
        Pass a pre-configured instance; `lmsafe.runner`'s experiment
        configs choose which one via `lmsafe.config`.
    lookback : int
        Window history used for the from-scratch retrain once drift
        fires.
    stat_fn : Callable[[np.ndarray], float]
        Reduces one window's per-edge score array to the scalar fed to
        the drift detector. Default: mean score.
    """

    name = "drift_triggered"

    def __init__(
        self,
        drift_detector: PageHinkley | SimplifiedADWIN,
        lookback: int = 168,
        stat_fn: Callable[[np.ndarray], float] = lambda s: float(np.mean(s)) if len(s) else 0.0,
    ):
        self.drift_detector = drift_detector
        self.lookback = lookback
        self.stat_fn = stat_fn
        self._history: deque[WindowGraph] = deque(maxlen=lookback)

    def fit_initial(self, detector: Detector, train_windows: Sequence[WindowGraph]) -> None:
        super().fit_initial(detector, train_windows)
        self._history.extend(train_windows)

    def observe(self, window: WindowGraph, scores: np.ndarray, detector: Detector) -> Detector:
        self._history.append(window)
        stat = self.stat_fn(scores)
        if self.drift_detector.update(stat):
            detector.fit(list(self._history), warm_start=False)
        return detector


# ---------------------------------------------------------------------------
# RQ3 proposed defense
# ---------------------------------------------------------------------------


class GatedPolicy(UpdatePolicy):
    """Score-gated, delayed-commit, multi-host-corroborated updating
    with a permanent clean-anchor memory.

    Three mechanisms, each targeting a specific weakness of the naive
    baselines above:

    1. **Score-gated admission.** An edge is only even a *candidate* for
       future training if its own anomaly score (from the detector's
       *current* model, computed before any update this window) is at
       or below the `score_admission_quantile` quantile of this
       window's scores -- i.e. the detector's current best guess is
       already "this looks fine". An edge that already looks highly
       suspicious is never used to teach the model that it is normal,
       no matter what else happens to it below.

    2. **Delayed commit with multi-host corroboration.** An admitted
       edge sits in a quarantine buffer for `quarantine_windows` windows
       before it is eligible to be promoted into the training buffer,
       and is only actually promoted if, within that same horizon, its
       *destination* host was also reached by at least
       `min_corroborating_src_hosts` *other, distinct* source hosts.
       This is directly motivated by this project's own inspection of
       the real LANL red-team file: 701 of 749 real lateral-movement
       events originate from a single source host. A conditioning
       attacker operating from one or a few hosts cannot manufacture
       corroboration from unrelated hosts without a much larger,
       easier-to-notice compromise -- whereas genuine benign drift
       (e.g. a newly provisioned subnet or file server) is, by the RQ1
       hypothesis, *collective*: many legitimate hosts reach it
       independently within a short horizon. Corroboration uses only
       structural agreement across hosts, never the ground-truth label.

    3. **Clean anchor memory.** The original benign training windows
       (`fit_initial`'s `train_windows`) are kept forever and mixed
       into *every* retrain alongside the promoted buffer. Even if some
       poisoned data slips past (1) and (2), the model is retrained
       from scratch each time on anchor-plus-promoted data, not warm-
       started from a possibly-already-biased running model -- bounding
       how far any slipped-through poisoning can pull the decision
       boundary in one retrain cycle.

    Parameters
    ----------
    score_admission_quantile : float in (0, 1]
        Edges scoring above this quantile of the window's own score
        distribution are never admitted to quarantine at all.
    quarantine_windows : int
        How long an admitted edge waits for corroboration before a
        promotion decision is made.
    min_corroborating_src_hosts : int
        Minimum distinct *other* source hosts that must also reach the
        same destination host within the quarantine horizon.
    retrain_period : int
        How often (in test windows) to actually retrain, from scratch,
        on anchor + promoted buffer.
    replay_cap : int
        Maximum number of promoted windows kept in the buffer (oldest
        dropped first) -- bounds memory on a long test stream.
    """

    name = "gated"

    def __init__(
        self,
        score_admission_quantile: float = 0.9,
        quarantine_windows: int = 24,
        min_corroborating_src_hosts: int = 3,
        retrain_period: int = 6,
        replay_cap: int = 500,
    ):
        if not 0 < score_admission_quantile <= 1:
            raise ValueError("score_admission_quantile must be in (0, 1]")
        self.score_admission_quantile = score_admission_quantile
        self.quarantine_windows = quarantine_windows
        self.min_corroborating_src_hosts = min_corroborating_src_hosts
        self.retrain_period = retrain_period
        self.replay_cap = replay_cap

        self._anchor_windows: list[WindowGraph] = []
        # Each entry: (admitted_at_window_idx, WindowGraph, admitted_edge_indices)
        self._quarantine: deque[tuple[int, WindowGraph, list[int]]] = deque()
        self._committed_buffer: deque[WindowGraph] = deque(maxlen=replay_cap)
        self._window_idx = 0
        self._since_retrain = 0

        # Diagnostics, useful for the paper's ablations/figures: how
        # much of what was *admitted* actually got *promoted*, per run.
        self.n_admitted_total = 0
        self.n_promoted_total = 0

    def fit_initial(self, detector: Detector, train_windows: Sequence[WindowGraph]) -> None:
        super().fit_initial(detector, train_windows)
        self._anchor_windows = list(train_windows)

    def observe(self, window: WindowGraph, scores: np.ndarray, detector: Detector) -> Detector:
        self._window_idx += 1

        admitted_idx = self._admit(window, scores)
        self.n_admitted_total += len(admitted_idx)
        self._quarantine.append((self._window_idx, window, admitted_idx))

        self._promote_matured_entries()

        self._since_retrain += 1
        if self._since_retrain >= self.retrain_period:
            self._since_retrain = 0
            train_batch = self._anchor_windows + list(self._committed_buffer)
            if train_batch:
                detector.fit(train_batch, warm_start=False)
        return detector

    # -- mechanism 1: score-gated admission ----------------------------

    def _admit(self, window: WindowGraph, scores: np.ndarray) -> list[int]:
        if len(scores) == 0:
            return []
        threshold = float(np.quantile(scores, self.score_admission_quantile))
        return [i for i, s in enumerate(scores) if s <= threshold]

    # -- mechanism 2: delayed commit + multi-host corroboration --------

    def _promote_matured_entries(self) -> None:
        while self._quarantine and self._window_idx - self._quarantine[0][0] >= self.quarantine_windows:
            cand_idx, cand_window, cand_edge_idx = self._quarantine.popleft()

            # Corroboration evidence: for every destination host, the
            # set of distinct source hosts that reached it in every
            # *other* entry still in the quarantine horizon (these span
            # [cand_idx+1, window_idx], i.e. "within the horizon after
            # this candidate was admitted" -- corroboration is allowed
            # to arrive any time up to the promotion decision).
            dst_to_srcs: dict[str, set] = defaultdict(set)
            for (_, w, eidx) in self._quarantine:
                for i in eidx:
                    a, b = w.edges[i]
                    dst_to_srcs[w.nodes[b]].add(w.nodes[a])

            kept = []
            for i in cand_edge_idx:
                a, b = cand_window.edges[i]
                src_host, dst_host = cand_window.nodes[a], cand_window.nodes[b]
                corroborators = dst_to_srcs.get(dst_host, set()) - {src_host}
                if len(corroborators) >= self.min_corroborating_src_hosts:
                    kept.append(i)

            if kept:
                self.n_promoted_total += len(kept)
                self._committed_buffer.append(filter_edges(cand_window, kept))
