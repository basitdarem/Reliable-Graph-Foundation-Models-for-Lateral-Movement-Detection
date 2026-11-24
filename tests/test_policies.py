import numpy as np

from lmsafe.config import SynthConfig
from lmsafe import synth
from lmsafe.data.windows import WindowGraph, iter_windows
from lmsafe.detectors import CountBaselineDetector
from lmsafe.drift import PageHinkley
from lmsafe.policies import (
    DriftTriggeredPolicy,
    FrozenPolicy,
    GatedPolicy,
    PeriodicPolicy,
    SlidingReplayPolicy,
)

from _doubles import RecordingDetector


def _synth_windows():
    cfg = SynthConfig(
        n_users=20, n_hosts=15, n_benign_days=10, base_edges_per_window=15,
        n_episodes=2, episode_chain_len=4, seed=5, drift_strength=0.2, drift_start_day=4,
    )
    events = synth.generate(cfg)
    windows = list(iter_windows(events, cfg.window_seconds))
    train_w = [w for w in windows if w.window_start < 41 * 3600]
    test_w = [w for w in windows if w.window_start >= 41 * 3600]
    return train_w, test_w


ALL_POLICIES = [
    lambda: FrozenPolicy(),
    lambda: PeriodicPolicy(period=12),
    lambda: SlidingReplayPolicy(window_size=48, retrain_period=12),
    lambda: DriftTriggeredPolicy(PageHinkley(delta=0.01, threshold=5.0), lookback=48),
    lambda: GatedPolicy(score_admission_quantile=0.9, quarantine_windows=12, min_corroborating_src_hosts=2, retrain_period=12),
]


def test_every_policy_runs_end_to_end_without_error():
    train_w, test_w = _synth_windows()
    for make_policy in ALL_POLICIES:
        det = CountBaselineDetector()
        pol = make_policy()
        pol.fit_initial(det, train_w)
        for w in test_w:
            s = det.score(w)
            assert np.all(np.isfinite(s))
            det = pol.observe(w, s, det)


def test_frozen_never_calls_fit_again_after_initial():
    train_w, _ = _synth_windows()
    det = RecordingDetector()
    pol = FrozenPolicy()
    pol.fit_initial(det, train_w)
    assert len(det.fit_calls) == 1
    for _ in range(50):
        w = WindowGraph(window_start=0, window_seconds=3600)
        det = pol.observe(w, np.zeros(0), det)
    assert len(det.fit_calls) == 1  # still just the initial fit


def test_periodic_retrains_every_period_windows_on_disjoint_batches():
    det = RecordingDetector()
    pol = PeriodicPolicy(period=3)  # lookback defaults to period -> non-overlapping
    pol.fit_initial(det, [])
    assert det.fit_calls == [(0, False)]

    empty_w = lambda i: WindowGraph(window_start=i, window_seconds=1)
    for i in range(9):
        det = pol.observe(empty_w(i), np.zeros(0), det)
    # One retrain every 3 windows -> 3 retrains, each on exactly 3 windows.
    assert det.fit_calls == [(0, False), (3, False), (3, False), (3, False)]


def test_sliding_replay_window_overlaps_across_retrains():
    det = RecordingDetector()
    pol = SlidingReplayPolicy(window_size=5, retrain_period=2)
    pol.fit_initial(det, [])
    empty_w = lambda i: WindowGraph(window_start=i, window_seconds=1)
    for i in range(6):
        det = pol.observe(empty_w(i), np.zeros(0), det)
    # buffer caps at 5; after 6 windows appended, retrains happen at
    # i=1 (buffer size 2), i=3 (buffer size 4), i=5 (buffer size 5, capped)
    assert det.fit_calls == [(0, False), (2, False), (4, False), (5, False)]


def test_periodic_and_sliding_are_not_equivalent_with_documented_defaults():
    """Regression test: these two policies must behave differently
    when PeriodicPolicy is used with its documented default (lookback
    == period, disjoint batches) even though both are internally
    implemented as a capped deque + periodic from-scratch refit."""
    train_w, test_w = _synth_windows()[0], _synth_windows()[1][:60]

    det_p = CountBaselineDetector()
    pol_p = PeriodicPolicy(period=12)
    pol_p.fit_initial(det_p, train_w)
    scores_p = []
    for w in test_w:
        s = det_p.score(w)
        scores_p.extend(s.tolist())
        det_p = pol_p.observe(w, s, det_p)

    det_s = CountBaselineDetector()
    pol_s = SlidingReplayPolicy(window_size=168, retrain_period=12)
    pol_s.fit_initial(det_s, train_w)
    scores_s = []
    for w in test_w:
        s = det_s.score(w)
        scores_s.extend(s.tolist())
        det_s = pol_s.observe(w, s, det_s)

    assert scores_p != scores_s


def test_drift_triggered_retrains_only_when_drift_detector_fires():
    det = RecordingDetector(score_fn=lambda w: np.zeros(len(w.edges)))
    # threshold absurdly high: PageHinkley should never fire on all-zero stats
    pol = DriftTriggeredPolicy(PageHinkley(delta=0.0, threshold=1e9), lookback=10)
    pol.fit_initial(det, [])
    empty_w = lambda i: WindowGraph(window_start=i, window_seconds=1)
    for i in range(20):
        s = det.score(empty_w(i))
        det = pol.observe(empty_w(i), s, det)
    assert len(det.fit_calls) == 1  # only the initial fit, never triggered


def test_gated_admission_excludes_high_scoring_edges():
    pol = GatedPolicy(score_admission_quantile=0.5)
    w = WindowGraph(
        window_start=0, window_seconds=3600,
        nodes=["A", "B", "C", "D"], node_index={"A": 0, "B": 1, "C": 2, "D": 3},
        edges=[(0, 1), (0, 2), (0, 3)], edge_weight=[1, 1, 1], edge_is_malicious=[False, False, False],
        n_events=3,
    )
    scores = np.array([1.0, 5.0, 9.0])  # median = 5.0
    admitted = pol._admit(w, scores)
    assert admitted == [0, 1]  # only scores <= median are admitted


def test_gated_promotes_only_corroborated_edges():
    """Hand-built scenario: a conditioning edge from a single source
    host is never corroborated by other hosts and must NOT be
    promoted; a destination independently reached by several other
    source hosts within the quarantine horizon must be promoted."""
    pol = GatedPolicy(
        score_admission_quantile=1.0,  # admit everything, isolate the corroboration step
        quarantine_windows=2,
        min_corroborating_src_hosts=2,
        retrain_period=1000,  # never auto-retrain; inspect the buffer directly
    )
    pol.fit_initial(CountBaselineDetector(), [])

    def w(i, edges):
        nodes = sorted({h for e in edges for h in e})
        idx = {h: j for j, h in enumerate(nodes)}
        return WindowGraph(
            window_start=i, window_seconds=1, nodes=nodes, node_index=idx,
            edges=[(idx[a], idx[b]) for a, b in edges],
            edge_weight=[1] * len(edges), edge_is_malicious=[False] * len(edges),
            n_events=len(edges),
        )

    det = CountBaselineDetector()
    scores_zero = lambda e: np.zeros(len(e.edges))

    # Window 0: attacker "ATTACKER" alone reaches "TARGET" (never corroborated).
    # Also "SRV" is reached by three different hosts across windows 0-1 (corroborated).
    det = pol.observe(w(0, [("ATTACKER", "TARGET"), ("H1", "SRV")]), scores_zero(w(0, [("ATTACKER", "TARGET"), ("H1", "SRV")])), det)
    det = pol.observe(w(1, [("H2", "SRV")]), scores_zero(w(1, [("H2", "SRV")])), det)
    det = pol.observe(w(2, [("H3", "SRV")]), scores_zero(w(2, [("H3", "SRV")])), det)
    # window_idx is now 3; the window-0 entry (admitted_at=1) matures once
    # 3 - 1 >= quarantine_windows(2), i.e. after this third observe() call.

    promoted_edges = set()
    for cand_window in pol._committed_buffer:
        for a, b in cand_window.edges:
            promoted_edges.add((cand_window.nodes[a], cand_window.nodes[b]))

    assert ("ATTACKER", "TARGET") not in promoted_edges
    assert ("H1", "SRV") in promoted_edges
