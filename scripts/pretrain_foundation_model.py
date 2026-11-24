#!/usr/bin/env python3
"""
Pretrain lmsafe.detectors.rw_gnn.GraphFoundationDetector's encoder +
edge-scorer weights on one network's benign windows, and save a
checkpoint that scripts/run_cross_network_transfer.py (or any other
script) can hand to a fresh `GraphFoundationDetector` via
`pretrained_state_dict=` for zero-shot or fine-tuned use on a
*different* network.

This is the "foundation model" step, scoped exactly as described in
lmsafe/detectors/rw_gnn.py's module docstring: a small, CPU-trainable
self-supervised link-prediction pretraining run, not a claim to
reproduce a large published architecture.

Sources
--------
--source synthetic
    Pretrain on lmsafe.synth's benign generator -- fast, no real data
    required, useful for development/dry runs and for the notebook's
    offline verification path.
--source lanl-cache --cache-path <path>
    Pretrain on a `(train_w, test_w)` pickle produced by
    `fetch_lanl_window.py build-cache` or `run_experiment.py`'s own
    `lanl.cache_path`. By default only `train_w` (the clean 41-hour
    training split) is used, specifically so the resulting checkpoint
    stays safe to later evaluate against that same network's *test*
    windows elsewhere without having pretrained on them first. Pass
    --include-test-windows to opt into pretraining on the full stream
    instead (only appropriate when this checkpoint will only ever be
    evaluated on a *different* network, never on this one's own test
    split).

Usage
------
    python scripts/pretrain_foundation_model.py --source synthetic \\
        --out checkpoints/foundation_synthetic.pt

    python scripts/pretrain_foundation_model.py --source lanl-cache \\
        --cache-path lmsafe_cache/lanl_full_windows.pkl \\
        --out checkpoints/foundation_lanl.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch

from lmsafe import synth
from lmsafe.config import SynthConfig
from lmsafe.data.windows import WindowGraph, iter_windows
from lmsafe.detectors.rw_gnn import GraphFoundationDetector


def _synthetic_windows(n_windows_target: int, seed: int) -> list[WindowGraph]:
    cfg = SynthConfig(
        n_users=200, n_hosts=150, n_benign_days=max(4, n_windows_target // 24 + 1),
        base_edges_per_window=40, n_episodes=0, seed=seed,
    )
    events = synth.generate(cfg, episodes=[])
    windows = list(iter_windows(events, cfg.window_seconds))
    return windows[:n_windows_target] if n_windows_target else windows


def _lanl_cache_windows(cache_path: Path, include_test_windows: bool) -> list[WindowGraph]:
    with open(cache_path, "rb") as fh:
        train_w, test_w = pickle.load(fh)
    windows = list(train_w)
    if include_test_windows:
        windows += list(test_w)
    return windows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "lanl-cache"], required=True)
    p.add_argument("--cache-path", help="required for --source lanl-cache")
    p.add_argument("--include-test-windows", action="store_true", help="see module docstring -- only safe if this checkpoint will never be evaluated on this same network's own test split")
    p.add_argument("--n-synthetic-windows", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--epochs-per-fit", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.source == "synthetic":
        windows = _synthetic_windows(a.n_synthetic_windows, a.seed)
    else:
        if not a.cache_path:
            p.error("--cache-path is required for --source lanl-cache")
        windows = _lanl_cache_windows(Path(a.cache_path), a.include_test_windows)

    if not windows:
        print("[pretrain] no windows to pretrain on -- nothing to do", file=sys.stderr)
        sys.exit(1)

    print(f"[pretrain] pretraining GraphFoundationDetector on {len(windows):,} windows "
          f"(source={a.source}, hidden_dim={a.hidden_dim})")

    det = GraphFoundationDetector(
        hidden_dim=a.hidden_dim, epochs_per_fit=a.epochs_per_fit, lr=a.lr, seed=a.seed,
    )

    t0 = time.time()
    # Fit window-by-window (rather than one big `fit(windows)` call) so
    # progress can be reported on a long real-data run -- the detector's
    # own `fit()` already accepts a full sequence, but chunking it here
    # costs nothing and gives visibility matching this project's other
    # long-running scripts (see lmsafe.data.lanl.stream_auth_events's
    # own progress-reporting rationale).
    for i, w in enumerate(windows, start=1):
        det.fit([w], warm_start=True)
        if i % a.progress_every == 0:
            print(f"[pretrain] fit {i:,}/{len(windows):,} windows, "
                  f"{det.state_summary()}, {time.time() - t0:.0f}s elapsed",
                  file=sys.stderr, flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(det.export_state_dict(), out)
    print(f"[pretrain] done in {time.time() - t0:.0f}s -- wrote checkpoint to {out}")
    print(f"[pretrain] final state: {det.state_summary()}")


if __name__ == "__main__":
    main()
