import numpy as np
import pytest

from lmsafe.config import SynthConfig
from lmsafe import synth
from lmsafe.data.windows import iter_windows
from lmsafe.detectors import CountBaselineDetector, FeatureMLPDetector, GraphFoundationDetector

DETECTOR_FACTORIES = [
    lambda: CountBaselineDetector(alpha=0.5),
    lambda: FeatureMLPDetector(epochs_per_fit=2, hidden_dim=8, seed=0),
    lambda: GraphFoundationDetector(epochs_per_fit=2, hidden_dim=8, seed=0),
]


def _windows():
    cfg = SynthConfig(
        n_users=15, n_hosts=12, n_benign_days=4, base_edges_per_window=10,
        n_episodes=0, seed=3,
    )
    events = synth.generate(cfg, episodes=[])
    return list(iter_windows(events, cfg.window_seconds))


@pytest.mark.parametrize("make_detector", DETECTOR_FACTORIES)
def test_score_is_finite_and_aligned_to_edges(make_detector):
    windows = _windows()
    det = make_detector()
    det.fit(windows[:20], warm_start=False)
    for w in windows[20:25]:
        scores = det.score(w)
        assert len(scores) == len(w.edges)
        assert np.all(np.isfinite(scores))


@pytest.mark.parametrize("make_detector", DETECTOR_FACTORIES)
def test_unseen_edge_scores_higher_than_a_frequently_seen_edge(make_detector):
    """Core sanity check of the score convention: a never-seen-before
    edge must score as MORE anomalous (higher) than an edge the
    detector was trained on heavily -- higher = more anomalous is the
    convention every policy/metric in this project relies on."""
    cfg = SynthConfig(
        n_users=6, n_hosts=5, n_benign_days=3, base_edges_per_window=30,
        n_episodes=0, seed=4,
    )
    events = synth.generate(cfg, episodes=[])
    windows = list(iter_windows(events, cfg.window_seconds))
    det = make_detector()
    det.fit(windows, warm_start=False)

    # Build one synthetic test window: one edge seen a lot during
    # training (if any), one edge guaranteed never seen (brand-new
    # host names).
    from lmsafe.data.windows import WindowGraph

    # pick the most frequent edge across training for the "familiar" case
    from collections import Counter

    counts = Counter()
    for w in windows:
        for (a, b), wt in zip(w.edges, w.edge_weight):
            counts[(w.nodes[a], w.nodes[b])] += wt
    familiar_src, familiar_dst = counts.most_common(1)[0][0]

    test_window = WindowGraph(
        window_start=10**9,
        window_seconds=cfg.window_seconds,
        nodes=[familiar_src, familiar_dst, "NEVER_SEEN_SRC", "NEVER_SEEN_DST"],
        node_index={familiar_src: 0, familiar_dst: 1, "NEVER_SEEN_SRC": 2, "NEVER_SEEN_DST": 3},
        edges=[(0, 1), (2, 3)],
        edge_weight=[1, 1],
        edge_is_malicious=[False, False],
        n_events=2,
    )
    scores = det.score(test_window)
    assert scores[1] > scores[0], (
        f"expected unseen edge to score higher (more anomalous) than a familiar edge, "
        f"got familiar={scores[0]}, unseen={scores[1]}"
    )


@pytest.mark.parametrize("make_detector", DETECTOR_FACTORIES)
def test_clone_fresh_is_independent_and_untrained(make_detector):
    windows = _windows()
    det = make_detector()
    det.fit(windows[:10], warm_start=False)
    fresh = det.clone_fresh()
    assert fresh.state_summary() != det.state_summary() or fresh.state_summary() == {}
    # Fitting the clone must not affect the original.
    before = det.state_summary()
    fresh.fit(windows[10:15], warm_start=False)
    assert det.state_summary() == before


@pytest.mark.parametrize("make_detector", DETECTOR_FACTORIES)
def test_warm_start_false_resets_prior_state(make_detector):
    windows = _windows()
    det = make_detector()
    det.fit(windows[:10], warm_start=False)
    summary_after_first_fit = dict(det.state_summary())
    det.fit(windows[10:12], warm_start=False)
    # A from-scratch refit on fewer windows should not retain the
    # larger edge/host vocabulary accumulated by the first fit.
    assert det.state_summary() != summary_after_first_fit


def test_detector_score_never_reads_ground_truth_label():
    """Static guard: no detector's score() method may reference
    edge_is_malicious (see lmsafe.data.schema's ground-truth boundary)."""
    import inspect

    from lmsafe.detectors import count_baseline, feature_mlp, rw_gnn

    for module in (count_baseline, feature_mlp, rw_gnn):
        src = inspect.getsource(module)
        assert "is_malicious" not in src, f"{module.__name__} must never read ground-truth labels"
