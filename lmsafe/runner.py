"""
Experiment orchestration: data -> policy loop -> detector -> metrics,
for one "experiment cell" (one detector x one policy x one dataset
variant), plus a grid runner that sweeps many cells into one tidy table.

This module intentionally knows nothing about *where* `train_windows`/
`test_windows` came from -- LANL, OpTC, or `lmsafe.synth` -- which is the
entire point of the `Event`/`WindowGraph` boundary described in
`lmsafe.data.schema`. `scripts/run_experiment.py` is the thin,
dataset-aware layer on top of this module that a YAML config selects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import pandas as pd

from .data.windows import WindowGraph
from .detectors.base import Detector
from .metrics import (
    WindowScores,
    cluster_episodes,
    detection_delay_per_episode,
    episode_bootstrap_ci,
    episode_hit,
    recall_at_budget,
    weighted_ap,
    weighted_auc,
)
from .policies import UpdatePolicy


@dataclass
class ExperimentResult:
    """Everything one (detector, policy, dataset-variant) cell produced.

    Kept as a dataclass (not just a dict) so `run_grid`'s DataFrame has
    a documented, stable set of columns -- a typo in a dict key used to
    build a results table is a silent empty column, not an error.
    """

    detector_name: str
    policy_name: str
    scenario: str
    weighted_auc: float
    weighted_ap: float
    rec_at_budget: dict[int, float]
    episode_hit_rate: float
    mean_delay_windows: Optional[float]
    delay_ci_lo: Optional[float]
    delay_ci_hi: Optional[float]
    n_episodes: int
    n_test_windows: int
    diagnostics: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flatten into one dict suitable for a pandas row -- one
        column per alert budget (`rec_at_1`, `rec_at_5`, ...)."""
        row = {
            "detector": self.detector_name,
            "policy": self.policy_name,
            "scenario": self.scenario,
            "weighted_auc": self.weighted_auc,
            "weighted_ap": self.weighted_ap,
            "episode_hit_rate": self.episode_hit_rate,
            "mean_delay_windows": self.mean_delay_windows,
            "delay_ci_lo": self.delay_ci_lo,
            "delay_ci_hi": self.delay_ci_hi,
            "n_episodes": self.n_episodes,
            "n_test_windows": self.n_test_windows,
        }
        for b, r in self.rec_at_budget.items():
            row[f"rec_at_{b}"] = r
        for k, v in self.diagnostics.items():
            row[f"diag_{k}"] = v
        return row


def run_experiment_cell(
    train_windows: Sequence[WindowGraph],
    test_windows: Sequence[WindowGraph],
    detector: Detector,
    policy: UpdatePolicy,
    *,
    detector_name: str,
    scenario: str = "default",
    budgets: Sequence[int] = (1, 5, 10, 20),
    gap_windows_for_episodes: int = 2,
    delay_budget: Optional[int] = None,
    delay_n_boot: int = 500,
    delay_seed: int = 0,
) -> ExperimentResult:
    """Run one detector+policy combination over one test stream and
    compute every metric this project reports.

    Parameters
    ----------
    train_windows, test_windows : Sequence[WindowGraph]
        From `lmsafe.data.lanl.split_train_test` (fed through
        `lmsafe.data.windows.iter_windows`) or the equivalent synthetic
        split in `scripts/run_experiment.py`.
    detector : Detector
        A fresh (or pre-configured) detector instance; `policy.fit_initial`
        is called on it here, so pass an unfitted instance.
    policy : UpdatePolicy
    detector_name, scenario : str
        Free-form labels copied into the result/row, e.g.
        `scenario="drift_strength=0.5"` or `scenario="conditioned"` --
        whatever a given RQ script is sweeping over.
    budgets : Sequence[int]
        Per-window alert budgets to report Rec@B for.
    delay_budget : int, optional
        Which budget's alert set to use for `detection_delay_per_episode`
        / `episode_hit`. Defaults to the middle value of `budgets`.
    """
    policy.fit_initial(detector, list(train_windows))

    records: list[WindowScores] = []
    for w in test_windows:
        scores = detector.score(w)
        records.append(WindowScores.from_window(w, scores))
        detector = policy.observe(w, scores, detector)

    auc = weighted_auc(records)
    ap = weighted_ap(records)
    rec_at_budget = {b: recall_at_budget(records, b) for b in budgets}

    episodes = cluster_episodes(records, gap_windows=gap_windows_for_episodes)
    db = delay_budget if delay_budget is not None else sorted(budgets)[len(budgets) // 2]
    hits = episode_hit(records, db, episodes)
    delays = [d for d in detection_delay_per_episode(records, db, episodes) if d is not None]

    hit_rate = (sum(hits) / len(hits)) if hits else float("nan")
    if delays:
        mean_delay, lo, hi = episode_bootstrap_ci(delays, n_boot=delay_n_boot, seed=delay_seed)
    else:
        mean_delay, lo, hi = float("nan"), float("nan"), float("nan")

    diagnostics = dict(detector.state_summary())
    diagnostics.update(
        {f"policy_{k}": v for k, v in vars(policy).items() if isinstance(v, (int, float, str))}
    )

    return ExperimentResult(
        detector_name=detector_name,
        policy_name=policy.name,
        scenario=scenario,
        weighted_auc=auc,
        weighted_ap=ap,
        rec_at_budget=rec_at_budget,
        episode_hit_rate=hit_rate,
        mean_delay_windows=mean_delay,
        delay_ci_lo=lo,
        delay_ci_hi=hi,
        n_episodes=len(episodes),
        n_test_windows=len(test_windows),
        diagnostics=diagnostics,
    )


def run_grid(
    cells: Sequence[dict],
    *,
    train_windows_fn: Callable[[dict], Sequence[WindowGraph]],
    test_windows_fn: Callable[[dict], Sequence[WindowGraph]],
    **common_kwargs,
) -> pd.DataFrame:
    """Run a list of experiment cells and return one tidy `DataFrame`.

    Parameters
    ----------
    cells : Sequence[dict]
        Each dict must have keys `"detector"` (a `Detector` instance,
        unfitted), `"detector_name"` (str), `"policy"` (an
        `UpdatePolicy` instance), and `"scenario_key"` (any hashable,
        passed to `train_windows_fn`/`test_windows_fn` to select which
        data variant this cell runs on -- e.g. a `drift_strength` value
        or `"conditioned"` vs `"clean"`). Any extra keys are copied into
        `scenario` as a human-readable label if `"scenario"` itself is
        not given explicitly.
    train_windows_fn, test_windows_fn : Callable[[dict], Sequence[WindowGraph]]
        Given one cell's dict, return that scenario's train/test window
        lists. Memoize inside your own closure if several cells share
        the same `scenario_key` and regenerating the data is expensive
        -- `run_grid` itself does not cache.
    **common_kwargs
        Forwarded to every `run_experiment_cell` call (e.g. `budgets=`).

    Returns
    -------
    pd.DataFrame
        One row per cell, columns as in `ExperimentResult.to_row`.
    """
    rows = []
    for cell in cells:
        train_w = train_windows_fn(cell)
        test_w = test_windows_fn(cell)
        scenario = cell.get("scenario", str(cell.get("scenario_key", "default")))
        result = run_experiment_cell(
            train_w,
            test_w,
            detector=cell["detector"],
            policy=cell["policy"],
            detector_name=cell["detector_name"],
            scenario=scenario,
            **common_kwargs,
        )
        rows.append(result.to_row())
    return pd.DataFrame(rows)
