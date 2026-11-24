"""
Common interface every detector implements.

`lmsafe.policies` is written entirely against this interface -- it never
inspects a detector's internals -- so any new detector (a future, heavier
CyberGFM/ULTRA-style model, say) can be dropped in by implementing these
four methods, and every update policy, drift detector, and metric in this
project keeps working unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..data.windows import WindowGraph


class Detector(ABC):
    """Scores the edges of a `WindowGraph` with an anomaly score, where
    **higher = more anomalous** (never a "normality probability" -- that
    inversion is a classic source of silent sign-flip bugs in anomaly
    detection code, so this project standardizes on one direction
    everywhere and `tests/test_detectors.py` checks it for every
    concrete detector).
    """

    @abstractmethod
    def fit(self, windows: Sequence[WindowGraph], *, warm_start: bool = True) -> None:
        """(Re-)fit the detector on a sequence of benign (or
        policy-admitted) window graphs.

        Parameters
        ----------
        windows : Sequence[WindowGraph]
            Training data, in time order. For a count-based detector
            this just accumulates edge counts. For the MLP detector this
            runs a few optimization epochs.
        warm_start : bool
            If True, continue from the detector's current parameters
            (an incremental update). If False, reset to a fresh,
            untrained state first (a full retrain from scratch on
            exactly `windows` and nothing else). Update policies in
            `lmsafe.policies` choose which one they need -- e.g. the
            "periodic retrain" policy calls `fit(..., warm_start=False)`
            on its current window buffer, while "sliding + replay" calls
            `fit(..., warm_start=True)` to keep adapting continuously.
        """

    @abstractmethod
    def score(self, window: WindowGraph) -> np.ndarray:
        """Return an anomaly score for every edge in `window`, aligned
        by index to `window.edges` (so `scores[i]` is the score for
        `window.edges[i]`). Must **not** read `window.edge_is_malicious`
        -- that field is ground truth for evaluation only (see
        `lmsafe.data.schema` module docstring); a unit test greps for
        this.
        """

    @abstractmethod
    def clone_fresh(self) -> "Detector":
        """Return a new, untrained instance with the same
        hyperparameters. Used by policies that need to train a
        brand-new model from scratch (e.g. the "frozen" baseline's
        initial fit, or a gated policy that wants a model fit only on
        its clean anchor memory with no history from a possibly-biased
        running model).
        """

    def state_summary(self) -> dict:
        """Optional: a small dict of diagnostic scalars (e.g. number of
        edges seen, vocabulary size) for logging. Default: empty.
        """
        return {}
