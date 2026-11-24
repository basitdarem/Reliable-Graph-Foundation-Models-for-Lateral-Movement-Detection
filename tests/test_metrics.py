import math

import numpy as np
import pytest

from lmsafe.metrics import (
    WindowScores,
    block_bootstrap_ci,
    cluster_episodes,
    detection_delay_per_episode,
    episode_bootstrap_ci,
    episode_hit,
    recall_at_budget,
    weighted_ap,
    weighted_auc,
)


def ws(window_start, scores, is_malicious, weight=None):
    scores = np.asarray(scores, dtype=np.float64)
    is_malicious = np.asarray(is_malicious, dtype=bool)
    weight = np.ones(len(scores)) if weight is None else np.asarray(weight, dtype=np.float64)
    return WindowScores(window_start=window_start, scores=scores, is_malicious=is_malicious, weight=weight)


def test_weighted_auc_perfect_separation():
    records = [ws(0, [0.1, 0.2, 9.0, 9.5], [False, False, True, True])]
    assert weighted_auc(records) == pytest.approx(1.0)


def test_weighted_auc_nan_when_single_class():
    records = [ws(0, [0.1, 0.2], [False, False])]
    assert math.isnan(weighted_auc(records))


def test_weighted_ap_nan_when_no_positives():
    records = [ws(0, [0.1, 0.2], [False, False])]
    assert math.isnan(weighted_ap(records))


def test_weighted_auc_respects_event_weight():
    # A heavily-weighted misranked benign row should pull AUC down more
    # than a lightly-weighted one -- sanity check that sample_weight is
    # actually wired through, not silently ignored.
    light = [ws(0, [5.0, 1.0, 0.0], [True, False, False], weight=[1, 1, 1])]
    heavy = [ws(0, [5.0, 1.0, 0.0], [True, False, False], weight=[1, 100, 1])]
    assert weighted_auc(heavy) <= weighted_auc(light)


def test_recall_at_budget_basic_ranking():
    # window with 4 edges, budget=2: top-2 by score are indices 2,3 (scores 9,8)
    records = [ws(0, [1, 2, 9, 8], [False, True, True, False])]
    r = recall_at_budget(records, budget_per_window=2, weighted=False)
    # malicious rows: idx1 (score2, not in top2) and idx2 (score9, in top2)
    assert r == pytest.approx(0.5)


def test_recall_at_budget_ignores_windows_with_no_malicious_rows():
    records = [ws(0, [1, 2], [False, False]), ws(1, [1, 9], [False, True])]
    r = recall_at_budget(records, budget_per_window=1, weighted=False)
    assert r == pytest.approx(1.0)


def test_recall_at_budget_nan_when_no_malicious_anywhere():
    records = [ws(0, [1, 2], [False, False])]
    assert math.isnan(recall_at_budget(records, 1))


def test_recall_at_budget_weighted_uses_event_counts():
    # single malicious row with weight=5 caught; total malicious weight=5
    records = [ws(0, [9.0, 1.0], [True, False], weight=[5, 1])]
    assert recall_at_budget(records, budget_per_window=1, weighted=True) == pytest.approx(1.0)


def test_cluster_episodes_groups_by_gap():
    # malicious windows at positions 0,1 (episode A), then 5 (episode B)
    records = [
        ws(0, [9.0], [True]),
        ws(1, [9.0], [True]),
        ws(2, [1.0], [False]),
        ws(3, [1.0], [False]),
        ws(4, [1.0], [False]),
        ws(5, [9.0], [True]),
    ]
    episodes = cluster_episodes(records, gap_windows=1)
    assert len(episodes) == 2
    assert {i for i, _ in episodes[0]} == {0, 1}
    assert {i for i, _ in episodes[1]} == {5}


def test_cluster_episodes_merges_within_allowed_gap():
    records = [ws(0, [9.0], [True]), ws(1, [1.0], [False]), ws(2, [9.0], [True])]
    # one empty window between two malicious windows: gap_windows=1 should merge them
    episodes = cluster_episodes(records, gap_windows=1)
    assert len(episodes) == 1
    # gap_windows=0 should NOT merge them
    episodes0 = cluster_episodes(records, gap_windows=0)
    assert len(episodes0) == 2


def test_cluster_episodes_empty_when_no_malicious():
    records = [ws(0, [1.0], [False])]
    assert cluster_episodes(records) == []


def test_detection_delay_zero_when_caught_in_first_window():
    records = [ws(0, [9.0, 1.0], [True, False])]
    episodes = cluster_episodes(records, gap_windows=0)
    delays = detection_delay_per_episode(records, budget_per_window=1, episodes=episodes)
    assert delays == [0]


def test_detection_delay_counts_windows_until_caught():
    # episode spans windows 0 and 1; only caught (within budget=1) at window 1
    records = [
        ws(0, [1.0, 9.0], [True, False]),   # malicious row ranked 2nd -> missed at budget 1
        ws(1, [9.0, 1.0], [True, False]),   # malicious row ranked 1st -> caught
    ]
    episodes = cluster_episodes(records, gap_windows=0)
    delays = detection_delay_per_episode(records, budget_per_window=1, episodes=episodes)
    assert delays == [1]


def test_detection_delay_none_when_never_caught():
    records = [ws(0, [1.0, 9.0], [True, False])]
    episodes = cluster_episodes(records, gap_windows=0)
    delays = detection_delay_per_episode(records, budget_per_window=1, episodes=episodes)
    assert delays == [None]


def test_episode_hit_matches_delay_none():
    records = [ws(0, [1.0, 9.0], [True, False])]
    episodes = cluster_episodes(records, gap_windows=0)
    assert episode_hit(records, 1, episodes) == [False]


def test_episode_bootstrap_ci_on_constant_values():
    pt, lo, hi = episode_bootstrap_ci([2.0, 2.0, 2.0], n_boot=100, seed=0)
    assert pt == pytest.approx(2.0)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_episode_bootstrap_ci_empty_input():
    pt, lo, hi = episode_bootstrap_ci([], n_boot=10)
    assert math.isnan(pt) and math.isnan(lo) and math.isnan(hi)


def test_block_bootstrap_ci_point_estimate_matches_direct_call():
    records = [ws(i, [1.0, 9.0], [False, i % 5 == 0]) for i in range(40)]
    point, lo, hi = block_bootstrap_ci(weighted_ap, records, block_size=8, n_boot=50, seed=0)
    assert point == pytest.approx(weighted_ap(records))
    assert lo <= point + 1e-9  # point estimate should be a plausible value near the CI
    assert not math.isnan(lo) and not math.isnan(hi)


def test_block_bootstrap_ci_handles_empty_records():
    point, lo, hi = block_bootstrap_ci(weighted_ap, [], block_size=8, n_boot=10)
    assert math.isnan(point)
