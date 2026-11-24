"""
Concept-drift detectors used only to *trigger* an update policy.

Critical design rule, stated once here and enforced by every caller in
`lmsafe.policies`: a drift detector in this module NEVER sees the
anomaly score of an individual edge, and is NEVER used as (part of) the
anomaly score itself. It only ever watches an aggregate, per-window
summary statistic (e.g. "mean anomaly score this window" or "fraction of
edges this window above the current alert threshold") and answers one
yes/no question: *has this summary statistic's distribution shifted
enough that the detector's model of 'normal' should probably be
refreshed?*

Keeping this boundary matters for RQ2: the attacker's conditioning edges
are specifically designed to look individually unremarkable while
shifting the model's notion of normal over many windows. A drift
detector that happened to be confused with the anomaly scorer itself
would make the attack trivially detectable by construction and defeat
the whole point of the experiment -- the "DriftTriggered" policy in
`lmsafe.policies` is a *baseline to be attacked*, not a safeguard, and
RQ2 is precisely the experiment that measures how easily it is fooled.

Two detectors are implemented:

PageHinkley     A standard, exact, small-state sequential test for a
                 one-sided shift in the mean of a scalar stream (Page,
                 1954; the incremental form used widely in concept-drift
                 literature, e.g. in the `river` package).
SimplifiedADWIN  A documented simplification of Bifet & Gavaldà's (2007)
                 ADWIN. See its class docstring for exactly what is
                 simplified and why, and what to swap in if exact
                 fidelity to the original paper is needed for a
                 camera-ready claim.
"""
from __future__ import annotations

import math
from collections import deque


class PageHinkley:
    """Incremental Page-Hinkley test for a one-sided increase in the
    mean of a scalar stream.

    Maintains a running mean `m_t` and the cumulative deviation
    `sum_t = sum_{i<=t}(x_i - m_i - delta)`, together with its
    running minimum `min_sum_t`. Signals a change when
    `sum_t - min_sum_t` exceeds `threshold`, then resets all state (the
    standard "detect-and-reset" variant, appropriate here because a
    caller that gets a drift signal is about to retrain on fresh data
    anyway, so there is no benefit to remembering pre-reset statistics).

    Parameters
    ----------
    delta : float
        Minimum magnitude of change considered meaningful (slack term);
        guards against alarming on infinitesimal drift.
    threshold : float
        Alarm threshold (often called "lambda" in the literature). Larger
        -> fewer false alarms, slower detection.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 10.0):
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._sum = 0.0
        self._min_sum = 0.0

    def update(self, x: float) -> bool:
        """Feed one new observation. Returns True exactly on the step a
        change is signaled (state is reset on that same call)."""
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._sum += x - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        if ph > self.threshold:
            self.reset()
            return True
        return False


class SimplifiedADWIN:
    """A documented simplification of ADWIN (Bifet & Gavaldà, 2007).

    What is kept from the original algorithm
    -------------------------------------------
    The core statistical idea: maintain a window of recent scalar
    observations, and at every step test whether splitting the window
    at some point produces two sub-windows whose means differ by more
    than a Hoeffding-bound-derived threshold (scaled by the observed
    range and corrected for multiple testing across candidate split
    points). If so, the data *before* that split is stale relative to
    the most recent behavior and is discarded.

    What is simplified
    ---------------------
    The original ADWIN stores the window as a list of exponentially-
    sized *buckets* (size 1, 2, 4, 8, ... up to a per-level cap `M`),
    which is what lets it provide effectively unbounded lookback in
    `O(log N)` memory via periodic bucket merging. This implementation
    instead keeps the raw observations in a plain bounded `deque` of at
    most `max_window` scalars, and tests candidate splits only at
    power-of-two offsets from the most recent observation (so still
    `O(log window)` candidate tests per update, just without the
    bucket-merge memory optimization). For this project's use case --
    one scalar summary per hourly window, with `max_window` set to a
    few hundred -- the memory this simplification would have saved is a
    few kilobytes, so the simplification costs nothing in practice
    while being far simpler to verify correct than the original
    bucket-list data structure.

    If a camera-ready version of the paper needs results that match the
    original ADWIN bit-for-bit, swap in `river.drift.ADWIN`, which
    implements the full bucket-list algorithm; it exposes the same
    `update(x) -> bool`-shaped interface used by `lmsafe.policies`, so
    no other code needs to change.

    Parameters
    ----------
    delta : float
        Confidence parameter (smaller -> more conservative, fewer false
        alarms). This is the same "delta" as in the original paper, not
        related to `PageHinkley.delta`.
    max_window : int
        Hard cap on retained history.
    min_window : int
        Do not attempt any cut test until at least this many
        observations have been seen (avoids spurious alarms on very
        short streams where the Hoeffding bound is meaningless).
    """

    def __init__(self, delta: float = 0.002, max_window: int = 400, min_window: int = 16):
        self.delta = delta
        self.max_window = max_window
        self.min_window = min_window
        self._window: deque[float] = deque(maxlen=max_window)

    def reset(self) -> None:
        self._window.clear()

    def update(self, x: float) -> bool:
        """Feed one new observation. Returns True exactly on the step a
        change is signaled; on that same call, stale history (everything
        before the detected cut point) is discarded from the window."""
        self._window.append(x)
        n = len(self._window)
        if n < self.min_window:
            return False

        values = list(self._window)
        lo = min(values)
        hi = max(values)
        value_range = max(hi - lo, 1e-9)

        # Candidate "recent" sizes: powers of two, smallest first, up to
        # just under the full window -- mirrors the exponential bucket
        # granularity of the original algorithm without its merge
        # machinery.
        candidates = []
        k = 1
        while k < n:
            candidates.append(k)
            k *= 2

        best_cut = None
        for k in candidates:
            n_recent = k
            n_old = n - k
            if n_old < 1:
                continue
            recent = values[n - k:]
            old = values[: n - k]
            mean_recent = sum(recent) / n_recent
            mean_old = sum(old) / n_old

            # Harmonic-mean effective sample size and Bonferroni
            # correction across the candidates tested this step, as in
            # the original ADWIN's cut test.
            m = 1.0 / (1.0 / n_old + 1.0 / n_recent)
            delta_prime = self.delta / max(len(candidates), 1)
            epsilon = value_range * math.sqrt(
                (1.0 / (2.0 * m)) * math.log(4.0 / delta_prime)
            )

            if abs(mean_recent - mean_old) > epsilon:
                # Prefer the candidate that discards the most stale
                # data (largest n_old among valid cuts); candidates are
                # scanned smallest-k-first so keep searching rather than
                # stopping at the first hit.
                if best_cut is None or n_old > best_cut:
                    best_cut = n_old

        if best_cut is not None:
            # Discard the stale prefix: keep only the most recent
            # (n - best_cut) observations.
            kept = values[best_cut:]
            self._window = deque(kept, maxlen=self.max_window)
            return True
        return False
