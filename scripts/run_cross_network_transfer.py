#!/usr/bin/env python3
"""
Cross-network transfer: does pretraining
`lmsafe.detectors.rw_gnn.GraphFoundationDetector` on one network's
benign traffic ("network A") help it detect lateral movement on a
*different* network ("network B") whose host population is completely
disjoint, compared to training on network B alone?

This is the experiment the original proposal title's "Cross-Network
Transfer" clause refers to. The default run here uses `lmsafe.synth`
for both networks (fast, offline, reproducible) with network B's host
and user *names* deliberately relabeled into an OpTC-style scheme
("SysClient0001" instead of LANL-style "C1") so the transfer is real,
not an artifact of two synthetic runs happening to share a naming
convention. Pass --network-a-cache-path / --network-b-cache-path to
point at real cached (train_w, test_w) pickles instead (e.g. a LANL
cache for A once an OpTC cache exists for B, or vice versa) -- the
evaluation logic below is dataset-agnostic by construction (same
WindowGraph boundary every reader in lmsafe.data produces).

Three regimes, evaluated on network B's test windows
-------------------------------------------------------
scratch
    A fresh `GraphFoundationDetector`, fit only on B's own training
    split (no exposure to A at all). The "no transfer" baseline.
zero_shot
    Pretrained on A, then applied to B's test stream with **no** fit on
    B's training split whatsoever -- the strongest possible transfer
    claim, if it works at all.
finetune
    Pretrained on A, then fine-tuned (`fit(..., warm_start=True)`) on
    B's own (small) training split before evaluation -- the realistic
    middle ground most deployments would actually use.

To isolate "what did the starting point buy you" from "how adaptively
does each regime behave during the test stream" (a separate, already-
covered question -- see RQ1), every regime is **frozen** after its one
initial fit: during the test stream each window is scored, then folded
into the detector's running host-stats via `observe()` (never a
gradient step -- see that method's docstring), for every regime alike.

Usage
------
    python scripts/run_cross_network_transfer.py [--out-dir results/cross_network_transfer]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lmsafe import synth
from lmsafe.config import LANL_TRAIN_CUTOFF_S, LanlConfig, SynthConfig
from lmsafe.data import lanl, optc
from lmsafe.data.windows import WindowGraph, iter_windows
from lmsafe.detectors.rw_gnn import GraphFoundationDetector
from lmsafe.metrics import (
    WindowScores,
    cluster_episodes,
    episode_hit,
    recall_at_budget,
    weighted_ap,
    weighted_auc,
)

BUDGET = 10


def _relabel(windows: list[WindowGraph], prefix: str) -> list[WindowGraph]:
    """Return a copy of `windows` with every host name replaced by a
    deterministic `{prefix}NNNN`-style name, so network B is provably
    disjoint from network A's naming scheme (never just "a different
    random seed that happens to also look like 'C123'") -- the same
    technique used in tests/test_rw_gnn.py to make the inductive-
    transfer claim mean something concrete."""
    rename: dict[str, str] = {}

    def relabel_one(host: str) -> str:
        if host not in rename:
            rename[host] = f"{prefix}{len(rename):04d}"
        return rename[host]

    out = []
    for w in windows:
        new_nodes = [relabel_one(h) for h in w.nodes]
        out.append(
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
    return out


def _load_cache_windows(cache_path: Path) -> tuple[list[WindowGraph], list[WindowGraph]]:
    with open(cache_path, "rb") as fh:
        train_w, test_w = pickle.load(fh)
    return list(train_w), list(test_w)


def _build_synthetic_network_a(seed: int) -> list[WindowGraph]:
    """Network A is pretraining material only -- benign traffic, no
    injected episodes, never itself evaluated."""
    cfg = SynthConfig(n_users=150, n_hosts=120, n_benign_days=12, base_edges_per_window=35, n_episodes=0, seed=seed)
    events = synth.generate(cfg, episodes=[])
    windows = list(iter_windows(events, cfg.window_seconds))
    return _relabel(windows, "C")  # LANL-style, explicit for symmetry with B's relabeling


def _build_synthetic_network_b(seed: int) -> tuple[list[WindowGraph], list[WindowGraph]]:
    """Network B is the transfer target: benign traffic plus a handful
    of injected lateral-movement episodes in the test portion, so
    transfer quality is actually measurable (AUC/AP/episode-hit-rate
    against a real, known-positive signal), with an OpTC-shaped naming
    scheme fully disjoint from network A's."""
    cfg = SynthConfig(n_users=150, n_hosts=120, n_benign_days=12, base_edges_per_window=35, n_episodes=6, seed=seed)
    events = synth.generate(cfg)
    # Same 41-hour train/test boundary used throughout this project
    # (LANL_TRAIN_CUTOFF_S) -- "B's own small training split" means the
    # same thing here as everywhere else, not an arbitrarily tinier cut
    # that would make the `scratch` baseline artificially starved.
    lanl_cfg = LanlConfig(train_cutoff_s=LANL_TRAIN_CUTOFF_S, end_s=cfg.n_benign_days * 86400)
    train_events, test_events_iter = lanl.split_train_test(iter(events), lanl_cfg)
    train_w = list(iter_windows(train_events, cfg.window_seconds))
    test_w = list(iter_windows(test_events_iter, cfg.window_seconds))
    return _relabel(train_w, "SysClient"), _relabel(test_w, "SysClient")


def _build_real_optc_network_b(
    paths: list[str], train_cutoff_s: int, window_seconds: int, only_internal_internal: bool,
    label_with_ground_truth: bool = False,
) -> tuple[list[WindowGraph], list[WindowGraph]]:
    """Build network B's train/test windows from real OpTC ecar-bro FLOW
    files via `lmsafe.data.optc.stream_optc_events`. No relabeling here
    -- real OpTC hostnames ("SysClient0351.systemia.com", ...) are
    already a completely different population from network A's LANL-
    style ("C123") names, so the cross-network-transfer claim is real by
    construction, not something this script needs to manufacture.

    Important honesty note: a single `ecar-bro/short` partition file (or
    a few) may contain *zero* internal-to-internal flows at all -- the
    real sample this reader was built against was entirely outbound
    traffic to external IPs (benign beaconing plus one C2-looking
    sequence), which `only_internal_internal=True` correctly drops in
    full. If that happens here, `main()` below reports it plainly rather
    than silently producing an empty, uninterpretable result -- try
    files from `ecar-bro/evaluation` (the actual red-team engagement
    day, across more than one host range) for files likely to contain
    genuine lateral-movement-relevant internal traffic.
    """
    cfg = optc.OptcConfig(
        window_seconds=window_seconds,
        train_cutoff_s=train_cutoff_s,
        only_internal_internal=only_internal_internal,
        label_with_ground_truth=label_with_ground_truth,
    )
    events = optc.stream_optc_events(paths, cfg)
    if not events:
        return [], []
    train_events, test_events_iter = lanl.split_train_test(iter(events), cfg)
    train_w = list(iter_windows(train_events, cfg.window_seconds))
    test_w = list(iter_windows(test_events_iter, cfg.window_seconds))
    return train_w, test_w


def _evaluate_frozen(det: GraphFoundationDetector, test_windows: list[WindowGraph]) -> list[WindowScores]:
    """Score every test window with weights frozen at whatever the
    caller already fit; fold each window into the running host-stats
    via `observe()` afterward (never a gradient step) so later windows
    benefit from having "seen" earlier hosts -- see the module
    docstring's "Three regimes" section for why this applies uniformly
    to every regime."""
    records = []
    for w in test_windows:
        if not w.edges:
            continue
        scores = det.score(w)
        records.append(WindowScores.from_window(w, scores))
        det.observe(w)
    return records


def _summarize(name: str, records: list[WindowScores]) -> dict:
    episodes = cluster_episodes(records, gap_windows=1)
    hits = episode_hit(records, BUDGET, episodes)
    return {
        "regime": name,
        "weighted_auc": weighted_auc(records),
        "weighted_ap": weighted_ap(records),
        f"rec_at_{BUDGET}": recall_at_budget(records, BUDGET),
        "episode_hit_rate": (sum(hits) / len(hits)) if hits else float("nan"),
        "n_episodes": len(episodes),
        "n_test_windows": len(records),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network-a-cache-path", help="path to a (train_w, test_w) pickle for network A; default: synthetic")
    p.add_argument("--network-b-cache-path", help="path to a (train_w, test_w) pickle for network B; default: synthetic")
    p.add_argument(
        "--network-b-optc-paths", nargs="+",
        help="one or more real OpTC ecar-bro FLOW files (.json or .json.gz) to use as network B instead "
             "of a cache or synthetic data",
    )
    p.add_argument("--network-b-optc-train-cutoff-s", type=int, default=3600,
                    help="train/test boundary in seconds from the first event in the given OpTC files "
                         "(default 1h -- OpTC samples span hours/days, not LANL's months)")
    p.add_argument("--network-b-optc-keep-external", action="store_true",
                    help="keep edges to unresolved (external) IPs as pseudo-hosts instead of dropping them "
                         "-- for inspecting what a file contains, not for a real transfer result")
    p.add_argument("--network-b-optc-label-ground-truth", action="store_true",
                    help="label events against lmsafe.data.optc_groundtruth (the red-team host+day ground "
                         "truth transcribed from OpTCRedTeamGroundTruth.pdf) instead of leaving every "
                         "event unlabeled. Turns the malicious-label metrics (AUC, AP, Rec@B, "
                         "episode_hit_rate) from NaN into real numbers -- but see that module's docstring "
                         "for the host+day-not-event-level caveat before reporting them as-is")
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--epochs-per-fit", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/cross_network_transfer")
    a = p.parse_args()

    if a.network_a_cache_path:
        windows_a_train, windows_a_test = _load_cache_windows(Path(a.network_a_cache_path))
        windows_a = windows_a_train + windows_a_test
    else:
        windows_a = _build_synthetic_network_a(a.seed)

    if a.network_b_optc_paths:
        b_train, b_test = _build_real_optc_network_b(
            a.network_b_optc_paths, a.network_b_optc_train_cutoff_s, 3600,
            only_internal_internal=not a.network_b_optc_keep_external,
            label_with_ground_truth=a.network_b_optc_label_ground_truth,
        )
        if not b_train and not b_test:
            print(
                "[cross_network_transfer] the given OpTC file(s) produced ZERO usable "
                "internal-to-internal windows for network B -- this is expected for a file "
                "that only contains outbound traffic to external IPs (e.g. the real "
                "ecar-bro/short sample this reader was built against: benign cert-check "
                "beaconing plus one C2-looking sequence, nothing host-to-host internal). "
                "Try a file from ecar-bro/evaluation instead, or pass "
                "--network-b-optc-keep-external to at least smoke-test the pipeline with "
                "external-IP pseudo-hosts (not a meaningful transfer result)."
            )
            return
    elif a.network_b_cache_path:
        b_train, b_test = _load_cache_windows(Path(a.network_b_cache_path))
    else:
        b_train, b_test = _build_synthetic_network_b(a.seed + 1)

    print(f"[cross_network_transfer] network A: {len(windows_a):,} windows (pretraining only)")
    print(f"[cross_network_transfer] network B: {len(b_train):,} train + {len(b_test):,} test windows")
    if a.network_b_optc_paths and not a.network_b_optc_label_ground_truth:
        print(
            "[cross_network_transfer] NOTE: ground-truth labeling is OFF (pass "
            "--network-b-optc-label-ground-truth to turn it on), so every metric below "
            "involving a malicious label (AUC, AP, Rec@B, episode_hit_rate) will read as "
            "NaN / 0 episodes -- this run still exercises the full real transfer mechanism "
            "(pretrain on A, apply to real B), just without a ground-truth score."
        )
    elif a.network_b_optc_paths and a.network_b_optc_label_ground_truth:
        print(
            "[cross_network_transfer] NOTE: ground-truth labeling is ON, joined against "
            "lmsafe.data.optc_groundtruth (host+day labels transcribed from "
            "OpTCRedTeamGroundTruth.pdf, covering 2019-09-23 to 2019-09-25 only -- events "
            "outside that date range, or between hosts never on the red team's list, are "
            "always unlabeled/benign). This is a host+day join, not LANL's event-level one: "
            "see that module's docstring before treating these numbers as directly "
            "comparable to the LANL results on equal footing."
        )

    # -- pretrain once on A ------------------------------------------------
    pretrainer = GraphFoundationDetector(hidden_dim=a.hidden_dim, epochs_per_fit=a.epochs_per_fit, seed=a.seed)
    pretrainer.fit(windows_a, warm_start=False)
    checkpoint = pretrainer.export_state_dict()
    print(f"[cross_network_transfer] pretrained on A: {pretrainer.state_summary()}")

    rows = []

    # -- scratch: no exposure to A at all -----------------------------------
    scratch = GraphFoundationDetector(hidden_dim=a.hidden_dim, epochs_per_fit=a.epochs_per_fit, seed=a.seed)
    scratch.fit(b_train, warm_start=False)
    rows.append(_summarize("scratch", _evaluate_frozen(scratch, b_test)))

    # -- zero_shot: pretrained on A, no fit on B's train split at all -------
    zero_shot = GraphFoundationDetector(hidden_dim=a.hidden_dim, epochs_per_fit=a.epochs_per_fit, seed=a.seed, pretrained_state_dict=checkpoint)
    rows.append(_summarize("zero_shot", _evaluate_frozen(zero_shot, b_test)))

    # -- finetune: pretrained on A, then fine-tuned on B's train split ------
    finetune = GraphFoundationDetector(hidden_dim=a.hidden_dim, epochs_per_fit=a.epochs_per_fit, seed=a.seed, pretrained_state_dict=checkpoint)
    finetune.fit(b_train, warm_start=True)
    rows.append(_summarize("finetune", _evaluate_frozen(finetune, b_test)))

    df = pd.DataFrame(rows)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_network_transfer_results.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"[cross_network_transfer] wrote {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["weighted_auc", "weighted_ap", "episode_hit_rate"]):
        ax.bar(df["regime"], df[metric])
        ax.set_title(metric)
        ax.set_ylim(bottom=0)
    fig.suptitle("Cross-network transfer: network A (pretrain) -> network B (evaluate)")
    fig.tight_layout()
    png_path = out_dir / "cross_network_transfer.png"
    fig.savefig(png_path, dpi=120)
    print(f"[cross_network_transfer] wrote {png_path}")


if __name__ == "__main__":
    main()
