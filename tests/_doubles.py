"""Tiny test doubles shared across policy/runner tests."""
from __future__ import annotations

import numpy as np

from lmsafe.detectors.base import Detector


class RecordingDetector(Detector):
    """A `Detector` that does no real scoring -- it just records every
    `fit` call (how many windows, whether warm_start) so a policy's
    update *schedule* can be tested precisely, independent of any real
    detector's own statistical behavior.

    `score` returns a fixed, caller-supplied array (or zeros) so tests
    can also control exactly what a policy's score-gated logic sees.
    """

    def __init__(self, score_fn=None):
        self.fit_calls: list[tuple[int, bool]] = []  # (n_windows, warm_start)
        self._score_fn = score_fn

    def fit(self, windows, *, warm_start: bool = True) -> None:
        self.fit_calls.append((len(list(windows)), warm_start))

    def score(self, window) -> np.ndarray:
        if self._score_fn is not None:
            return self._score_fn(window)
        return np.zeros(len(window.edges), dtype=np.float64)

    def clone_fresh(self) -> "RecordingDetector":
        return RecordingDetector(self._score_fn)
