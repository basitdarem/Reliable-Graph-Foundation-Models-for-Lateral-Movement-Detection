"""Tests for lmsafe.runner: the detector/policy/metrics glue.

Uses `RecordingDetector` with a caller-supplied `score_fn` so every
score the metrics see is exactly known -- this lets the expected
weighted_auc / rec_at_budget / episode_hit_rate / delay values be
worked out by hand instead of merely checking "it runs".
"""
import math

import numpy as np
import pytest

from lmsafe.data.windows import WindowGraph
from lmsafe.policies import FrozenPolicy, PeriodicPolicy
from lmsafe.runner import ExperimentResult, run_experiment_cell, run_grid

from _doubles import RecordingDetector


def _window(window_start, malicious, scores):
    """Two-edge window: edges[(0,1), (1,0)] over nodes h0,h1.

    `malicious` and `scores` are length-2 sequences aligned to the
    edges above.
    """
    return WindowGraph(
        window_start=window_start,
        window_seconds=3600,
        nodes=["h0", "h1"],
        node_index={"h0": 0, "h1": 1},
        edges=[(0, 1), (1, 0)],
        edge_weight=[1, 1],
        edge_is_malicious=list(malicious),
        n_events=2,
    )


def _score_fn_table(table):
    """table: dict[window_start] -> list[float] of per-edge scores."""

    def _fn(window):
        return np.asarray(table[window.window_start], dtype=np.float64)

    return _fn


def _fixed_scenario():
    """Three test windows, one malicious edge total, deterministically
    ranked so recall/delay/hit-rate are all hand-checkable:

    w0 (t=0):    both benign,       scores [1.0, 0.0]
    w1 (t=3600): edge0 malicious,   scores [9.0, 0.0]  -> malicious edge ranks #1
    w2 (t=7200): both benign,       scores [1.0, 0.0]
    """
    train_windows = [_window(-7200, [False, False], [0.0, 0.0])]
    test_windows = [
        _window(0, [False, False], [1.0, 0.0]),
        _window(3600, [True, False], [9.0, 0.0]),
        _window(7200, [False, False], [1.0, 0.0]),
    ]
    scores_by_start = {0: [1.0, 0.0], 3600: [9.0, 0.0], 7200: [1.0, 0.0]}
    return train_windows, test_windows, scores_by_start


def test_run_experiment_cell_basic_metrics():
    train_w, test_w, score_table = _fixed_scenario()
    det = RecordingDetector(score_fn=_score_fn_table(score_table))
    pol = FrozenPolicy()

    result = run_experiment_cell(
        train_w,
        test_w,
        det,
        pol,
        detector_name="recording",
        scenario="unit_test",
        budgets=(1, 2),
        gap_windows_for_episodes=0,
        delay_budget=1,
    )

    assert isinstance(result, ExperimentResult)
    assert result.detector_name == "recording"
    assert result.policy_name == pol.name
    assert result.scenario == "unit_test"
    assert result.n_test_windows == 3
    assert result.n_episodes == 1  # one malicious window, isolated

    # Rec@1: in w1 the malicious edge is top-ranked -> caught; no
    # malicious rows in w0/w2, so those windows don't count.
    assert result.rec_at_budget[1] == pytest.approx(1.0)
    assert result.rec_at_budget[2] == pytest.approx(1.0)

    # Only one episode, caught in its very first (only) window -> delay 0.
    assert result.episode_hit_rate == pytest.approx(1.0)
    assert result.mean_delay_windows == pytest.approx(0.0)

    # AUC: pooled across all 6 edge-rows, malicious row scores 9.0 which
    # is the global max -> perfect separation.
    assert result.weighted_auc == pytest.approx(1.0)
    assert not math.isnan(result.weighted_ap)


def test_run_experiment_cell_calls_fit_initial_once():
    train_w, test_w, score_table = _fixed_scenario()
    det = RecordingDetector(score_fn=_score_fn_table(score_table))
    pol = FrozenPolicy()
    run_experiment_cell(train_w, test_w, det, pol, detector_name="recording")
    # FrozenPolicy.fit_initial calls detector.fit exactly once, and
    # FrozenPolicy.observe never refits -- this is the same contract
    # test_policies.py checks directly, exercised here through the runner.
    assert len(det.fit_calls) == 1
    assert det.fit_calls[0] == (len(train_w), False)


def test_run_experiment_cell_missed_detection_increases_delay_and_drops_recall():
    # Same shape as the fixed scenario, but the malicious edge is now
    # ranked LAST in its window, and budget=1 can't catch it there,
    # only in a later (here: nonexistent) window -> never caught.
    train_windows = [_window(-7200, [False, False], [0.0, 0.0])]
    test_windows = [
        _window(0, [True, False], [0.0, 9.0]),  # malicious edge ranks #2 of 2
    ]
    score_table = {0: [0.0, 9.0]}
    det = RecordingDetector(score_fn=_score_fn_table(score_table))
    pol = FrozenPolicy()

    result = run_experiment_cell(
        train_windows,
        test_windows,
        det,
        pol,
        detector_name="recording",
        budgets=(1,),
        gap_windows_for_episodes=0,
        delay_budget=1,
    )
    assert result.rec_at_budget[1] == pytest.approx(0.0)
    assert result.episode_hit_rate == pytest.approx(0.0)
    # never caught -> no finite delay to average
    assert math.isnan(result.mean_delay_windows)


def test_to_row_flattens_rec_at_budget_and_diagnostics():
    train_w, test_w, score_table = _fixed_scenario()
    det = RecordingDetector(score_fn=_score_fn_table(score_table))
    # PeriodicPolicy (unlike FrozenPolicy) has scalar instance attrs
    # (period, lookback, _since_retrain), which is what exercises the
    # diag_* namespacing in ExperimentResult.to_row.
    pol = PeriodicPolicy(period=1)
    result = run_experiment_cell(
        train_w, test_w, det, pol, detector_name="recording", budgets=(1, 5)
    )
    row = result.to_row()
    assert row["detector"] == "recording"
    assert row["policy"] == pol.name
    assert "rec_at_1" in row and "rec_at_5" in row
    assert row["rec_at_1"] == pytest.approx(result.rec_at_budget[1])
    assert row["rec_at_5"] == pytest.approx(result.rec_at_budget[5])
    # diagnostics (policy's own scalar attrs) are namespaced diag_*
    assert any(k.startswith("diag_") for k in row)
    assert row.get("diag_policy_period") == 1


def test_run_grid_produces_one_row_per_cell():
    train_w, test_w, score_table = _fixed_scenario()
    score_fn = _score_fn_table(score_table)

    cells = [
        {
            "detector": RecordingDetector(score_fn=score_fn),
            "detector_name": "recording",
            "policy": FrozenPolicy(),
            "scenario_key": "scenario_a",
        },
        {
            "detector": RecordingDetector(score_fn=score_fn),
            "detector_name": "recording",
            "policy": FrozenPolicy(),
            "scenario_key": "scenario_b",
        },
    ]

    df = run_grid(
        cells,
        train_windows_fn=lambda cell: train_w,
        test_windows_fn=lambda cell: test_w,
        budgets=(1,),
    )
    assert len(df) == 2
    assert set(df["scenario"]) == {"scenario_a", "scenario_b"}
    assert "rec_at_1" in df.columns
    # both cells run the identical fixed scenario -> identical AUC
    assert df["weighted_auc"].nunique() == 1


def test_run_grid_respects_explicit_scenario_label_over_key():
    train_w, test_w, score_table = _fixed_scenario()
    score_fn = _score_fn_table(score_table)
    cells = [
        {
            "detector": RecordingDetector(score_fn=score_fn),
            "detector_name": "recording",
            "policy": FrozenPolicy(),
            "scenario_key": "ignored_key",
            "scenario": "explicit_label",
        }
    ]
    df = run_grid(
        cells,
        train_windows_fn=lambda cell: train_w,
        test_windows_fn=lambda cell: test_w,
        budgets=(1,),
    )
    assert df.iloc[0]["scenario"] == "explicit_label"
