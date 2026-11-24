"""Dedicated tests for lmsafe.detectors.rw_gnn.GraphFoundationDetector,
beyond the shared Detector-contract tests in test_detectors.py.

These check the two things that are specific to *this* detector and not
covered by the generic parametrized suite: (1) the neighborhood
aggregation actually changes scores relative to not having it (i.e. this
isn't secretly just FeatureMLPDetector under another name), and (2) a
pretrained checkpoint genuinely transfers to a network with a completely
disjoint host-naming scheme -- the whole point of building this detector
for the cross-network-transfer research question.
"""
import numpy as np
import torch

from lmsafe.config import SynthConfig
from lmsafe import synth
from lmsafe.data.windows import WindowGraph, iter_windows
from lmsafe.detectors.rw_gnn import GraphFoundationDetector, _GraphEncoder
from lmsafe.detectors.feature_mlp import HOST_FEATURE_DIM


def _windows(seed=3, n_hosts=12):
    cfg = SynthConfig(
        n_users=15, n_hosts=n_hosts, n_benign_days=4, base_edges_per_window=10,
        n_episodes=0, seed=seed,
    )
    events = synth.generate(cfg, episodes=[])
    return list(iter_windows(events, cfg.window_seconds))


def test_encoder_neighborhood_aggregation_is_not_a_no_op():
    """The whole reason this is a GNN and not just FeatureMLPDetector
    again: a node's encoded representation must depend on its
    neighbors' features, not only its own. Build two tiny windows with
    IDENTICAL node 0 but different neighbors, and confirm node 0's
    post-encoder representation differs between them."""
    torch.manual_seed(0)
    encoder = _GraphEncoder(HOST_FEATURE_DIM, hidden_dim=8)

    h0_a = torch.zeros(3, HOST_FEATURE_DIM)
    h0_a[1] = 1.0  # node 0's only neighbor has feature vector of all-ones
    edge_index_a = torch.tensor([[0], [1]], dtype=torch.long)

    h0_b = torch.zeros(3, HOST_FEATURE_DIM)
    h0_b[2] = 5.0  # same graph shape, but node 0's neighbor feature is different
    edge_index_b = torch.tensor([[0], [2]], dtype=torch.long)

    out_a = encoder(h0_a, edge_index_a)
    out_b = encoder(h0_b, edge_index_b)
    assert not torch.allclose(out_a[0], out_b[0]), (
        "node 0's representation did not change when its neighbor's "
        "feature changed -- the neighborhood aggregation is a no-op"
    )


def test_fit_and_score_runs_on_synthetic_windows():
    windows = _windows()
    det = GraphFoundationDetector(epochs_per_fit=2, hidden_dim=8, seed=0)
    det.fit(windows[:15], warm_start=False)
    scores = det.score(windows[15])
    assert len(scores) == len(windows[15].edges)
    assert np.all(np.isfinite(scores))


def test_export_and_load_state_dict_round_trips_identical_scores():
    windows = _windows()
    det = GraphFoundationDetector(epochs_per_fit=2, hidden_dim=8, seed=1)
    det.fit(windows[:15], warm_start=False)
    scores_before = det.score(windows[15])

    checkpoint = det.export_state_dict()
    fresh = GraphFoundationDetector(
        epochs_per_fit=2, hidden_dim=8, seed=1, pretrained_state_dict=checkpoint
    )
    # No fit() on `fresh` -- this is exactly the zero-shot transfer path:
    # load pretrained weights, score immediately.
    scores_after = fresh.score(windows[15])
    # `fresh` has not accumulated the running host-stats history that
    # `det` built up over windows[:15], so scores will legitimately
    # differ in general (different _HostStats) -- but feeding the same
    # fresh host-stats state should reproduce `det`'s own first-window
    # behavior exactly, which is what this test actually checks: replay
    # the SAME fit history on `fresh` and confirm bit-for-bit identical
    # scoring, proving the checkpoint carries all the model weights
    # (nothing silently left out of export_state_dict).
    fresh2 = GraphFoundationDetector(
        epochs_per_fit=2, hidden_dim=8, seed=1, pretrained_state_dict=checkpoint
    )
    fresh2._stats = det._stats
    fresh2._edge_counts = dict(det._edge_counts)
    fresh2._window_idx = det._window_idx
    scores_replayed = fresh2.score(windows[15])
    np.testing.assert_allclose(scores_before, scores_replayed, rtol=1e-6)
    assert np.all(np.isfinite(scores_after))


def test_freeze_encoder_prevents_encoder_weights_from_changing():
    windows = _windows()
    pretrainer = GraphFoundationDetector(epochs_per_fit=2, hidden_dim=8, seed=2)
    pretrainer.fit(windows[:10], warm_start=False)
    checkpoint = pretrainer.export_state_dict()
    encoder_before = {k: v.clone() for k, v in checkpoint["encoder"].items()}

    frozen_det = GraphFoundationDetector(
        epochs_per_fit=3, hidden_dim=8, seed=2,
        pretrained_state_dict=checkpoint, freeze_encoder=True,
    )
    frozen_det.fit(windows[10:20], warm_start=True)
    encoder_after = frozen_det._encoder.state_dict()
    for k in encoder_before:
        assert torch.equal(encoder_before[k], encoder_after[k]), (
            f"encoder parameter {k} changed despite freeze_encoder=True"
        )


def test_observe_updates_stats_without_touching_weights():
    """`observe()` is the zero-shot-evaluation primitive: it must grow
    the running host-stats/edge-count state exactly like `fit()` does,
    but must never take a gradient step -- the whole point is "let the
    detector build local history on a new network without training on
    it"."""
    windows = _windows()
    det = GraphFoundationDetector(epochs_per_fit=2, hidden_dim=8, seed=0)
    weights_before = {k: v.clone() for k, v in det._encoder.state_dict().items()}

    det.observe(windows[0])
    det.observe(windows[1])

    assert det.state_summary()["n_windows_fit"] == 2
    assert det.state_summary()["n_hosts_seen"] > 0
    weights_after = det._encoder.state_dict()
    for k in weights_before:
        assert torch.equal(weights_before[k], weights_after[k]), (
            f"observe() changed encoder parameter {k}, but it must only "
            "update running stats, never train"
        )


def test_pretrained_weights_transfer_to_a_disjoint_host_population():
    """The core cross-network-transfer claim, exercised in miniature
    without real OpTC data: pretrain on a "network A" whose hosts are
    named one way, then apply those exact weights (no further fit) to a
    "network B" whose hosts have a completely disjoint naming scheme --
    simulating LANL-style "C1234" names versus OpTC-style
    "SysClient0051" names -- and confirm scoring still produces finite,
    sensible output. This is the property that would be impossible for
    a per-host-ID-embedding model (the CyberGFM failure mode this
    project's literature review flags) but is exactly what the
    feature-based, ID-free design here is for.
    """
    windows_a = _windows(seed=10, n_hosts=10)  # "network A", hosts named by synth.generate
    pretrainer = GraphFoundationDetector(epochs_per_fit=2, hidden_dim=8, seed=5)
    pretrainer.fit(windows_a[:20], warm_start=False)
    checkpoint = pretrainer.export_state_dict()

    # Build "network B": same synthetic generator, different seed (so a
    # disjoint random host/user population), then relabel every host
    # with an OpTC-shaped name to make the disjointness explicit and
    # mechanical, not just "different random strings that happen to
    # also look like LANL hosts".
    windows_b_raw = _windows(seed=99, n_hosts=10)
    rename = {}

    def relabel(host: str) -> str:
        if host not in rename:
            rename[host] = f"SysClient{len(rename):04d}"
        return rename[host]

    windows_b = []
    for w in windows_b_raw:
        new_nodes = [relabel(h) for h in w.nodes]
        windows_b.append(
            WindowGraph(
                window_start=w.window_start,
                window_seconds=w.window_seconds,
                nodes=new_nodes,
                node_index={h: i for i, h in enumerate(new_nodes)},
                edges=list(w.edges),
                edge_weight=list(w.edge_weight),
                edge_is_malicious=list(w.edge_is_malicious),
                n_events=w.n_events,
            )
        )
    assert not (set(w for win in windows_a for w in win.nodes) & set(w for win in windows_b for w in win.nodes)), (
        "test setup bug: network A and B host names must be fully disjoint"
    )

    transferred = GraphFoundationDetector(
        epochs_per_fit=2, hidden_dim=8, seed=5, pretrained_state_dict=checkpoint
    )
    # Zero-shot: no fit() at all on network B, just score directly with
    # the pretrained-on-A weights and B's own (freshly initialized,
    # cold-start) running host stats.
    scores = transferred.score(windows_b[5])
    assert len(scores) == len(windows_b[5].edges)
    assert np.all(np.isfinite(scores)), "pretrained weights produced non-finite scores on a disjoint host population"
