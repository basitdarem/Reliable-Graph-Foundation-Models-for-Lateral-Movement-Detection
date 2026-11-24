"""
Name -> object factories for detectors, policies, and drift detectors.

Exists so a YAML experiment config (`scripts/run_experiment.py`) or a
quick interactive session can build a detector/policy by name and a
dict of keyword arguments, instead of every caller importing every
concrete class directly. Every RQ script in `scripts/` also uses these,
so the set of valid names in one place (here) is the single source of
truth for what a config file is allowed to say.
"""
from __future__ import annotations

from typing import Any

from .detectors.base import Detector
from .detectors.count_baseline import CountBaselineDetector
from .detectors.feature_mlp import FeatureMLPDetector
from .detectors.rw_gnn import GraphFoundationDetector
from .drift import PageHinkley, SimplifiedADWIN
from .policies import (
    DriftTriggeredPolicy,
    FrozenPolicy,
    GatedPolicy,
    PeriodicPolicy,
    SlidingReplayPolicy,
    UpdatePolicy,
)

DETECTORS = {
    "count_baseline": CountBaselineDetector,
    "feature_mlp": FeatureMLPDetector,
    "graph_foundation": GraphFoundationDetector,
}

DRIFT_DETECTORS = {
    "page_hinkley": PageHinkley,
    "adwin": SimplifiedADWIN,
}


def make_detector(name: str, **kwargs: Any) -> Detector:
    try:
        cls = DETECTORS[name]
    except KeyError:
        raise ValueError(f"unknown detector {name!r}, expected one of {sorted(DETECTORS)}") from None
    return cls(**kwargs)


def make_drift_detector(name: str, **kwargs: Any):
    try:
        cls = DRIFT_DETECTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown drift detector {name!r}, expected one of {sorted(DRIFT_DETECTORS)}"
        ) from None
    return cls(**kwargs)


def make_policy(name: str, **kwargs: Any) -> UpdatePolicy:
    """Build a policy by name. `drift_triggered` additionally requires
    `drift_detector_name` (+ optional `drift_detector_kwargs`) in
    `kwargs`, consumed here rather than passed through to
    `DriftTriggeredPolicy.__init__`, which expects an already-built
    drift-detector instance.
    """
    if name == "frozen":
        return FrozenPolicy(**kwargs)
    if name == "periodic":
        return PeriodicPolicy(**kwargs)
    if name == "sliding_replay":
        return SlidingReplayPolicy(**kwargs)
    if name == "drift_triggered":
        kwargs = dict(kwargs)
        dd_name = kwargs.pop("drift_detector_name", "page_hinkley")
        dd_kwargs = kwargs.pop("drift_detector_kwargs", {})
        drift_detector = make_drift_detector(dd_name, **dd_kwargs)
        return DriftTriggeredPolicy(drift_detector=drift_detector, **kwargs)
    if name == "gated":
        return GatedPolicy(**kwargs)
    raise ValueError(
        f"unknown policy {name!r}, expected one of "
        "['frozen', 'periodic', 'sliding_replay', 'drift_triggered', 'gated']"
    )
