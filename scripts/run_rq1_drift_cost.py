#!/usr/bin/env python3
"""
RQ1 -- Drift cost: how much detection quality do different update
policies lose to *benign* drift alone, with no attacker conditioning at
all, and how fast do they recover?

Experiment design
-------------------
A single, fixed set of lateral-movement episodes (same actor hosts,
same target chains, same timing) is injected into four otherwise-
identical synthetic benign backgrounds that differ only in
`drift_strength` (0.0 = perfectly stationary, up to 1.0 = strong,
persistent population growth starting on `DRIFT_START_DAY`). Holding
the attack fixed and varying only the benign background isolates "cost
of drift" from "how hard this particular attack is to find" -- the
counterfactual RQ1 needs and that the real, single-recording LANL
dataset cannot offer on its own (see `lmsafe.synth` module docstring).

Every policy in `lmsafe.policies` is run against every drift level with
the `CountBaselineDetector` (fast enough for a full sweep on a 2-CPU
sandbox); `FeatureMLPDetector` is additionally spot-checked at the
strongest drift level only, as a sanity check that the finding is not
an artifact of the particular detector (a full sweep with it is left to
`scripts/run_experiment.py` + a YAML config, which can be pointed at
either detector for the real LANL/OpTC runs).

Usage
------
    python scripts/run_rq1_drift_cost.py [--out-dir results/rq1]

Outputs `<out-dir>/rq1_results.csv` and `<out-dir>/rq1_drift_cost.png`.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lmsafe import synth
from lmsafe.config import LANL_TRAIN_CUTOFF_S, SynthConfig
from lmsafe.data.windows import iter_windows
from lmsafe.detectors import CountBaselineDetector, FeatureMLPDetector
from lmsafe.policies import (
    DriftTriggeredPolicy,
    FrozenPolicy,
    GatedPolicy,
    PeriodicPolicy,
    SlidingReplayPolicy,
)
from lmsafe.drift import PageHinkley
from lmsafe.runner import run_experiment_cell

DRIFT_LEVELS = [0.0, 0.3, 0.6, 1.0]
DRIFT_START_DAY = 6
N_BENIGN_DAYS = 25
N_EPISODES = 5
EPISODE_CHAIN_LEN = 5
SEED = 42
BUDGETS = (1, 5, 10, 20)


def base_cfg(drift_strength: float) -> SynthConfig:
    return SynthConfig(
        n_users=40,
        n_hosts=25,
        n_benign_days=N_BENIGN_DAYS,
        base_edges_per_window=25,
        n_episodes=N_EPISODES,
        episode_chain_len=EPISODE_CHAIN_LEN,
        drift_start_day=DRIFT_START_DAY,
        drift_strength=drift_strength,
        seed=SEED,
    )


def fixed_episodes() -> list[synth.LMEpisode]:
    """Episodes placed once, on the drift_strength=0 population, and
    reused unchanged across every drift level (see module docstring)."""
    import numpy as np

    cfg0 = base_cfg(0.0)
    rng = np.random.default_rng(cfg0.seed)
    pop = synth.Population(cfg0, rng)
    return synth.default_episodes(cfg0, pop, rng)


def split(cfg: SynthConfig, episodes):
    events = synth.generate(cfg, episodes=episodes)
    windows = list(iter_windows(events, cfg.window_seconds))
    train_w = [w for w in windows if w.window_start < LANL_TRAIN_CUTOFF_S]
    test_w = [w for w in windows if w.window_start >= LANL_TRAIN_CUTOFF_S]
    return train_w, test_w


def policy_factory(name: str):
    if name == "frozen":
        return FrozenPolicy()
    if name == "periodic":
        return PeriodicPolicy(period=24)  # lookback defaults to period: disjoint, non-overlapping batches
    if name == "sliding_replay":
        return SlidingReplayPolicy(window_size=168, retrain_period=24)
    if name == "drift_triggered":
        return DriftTriggeredPolicy(PageHinkley(delta=0.01, threshold=8.0), lookback=168)
    if name == "gated":
        return GatedPolicy(
            score_admission_quantile=0.9,
            quarantine_windows=24,
            min_corroborating_src_hosts=3,
            retrain_period=24,
        )
    raise ValueError(name)


POLICY_NAMES = ["frozen", "periodic", "sliding_replay", "drift_triggered", "gated"]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = fixed_episodes()
    print(f"[rq1] {len(episodes)} fixed lateral-movement episodes, reused at every drift level")

    rows = []
    for drift in DRIFT_LEVELS:
        cfg = base_cfg(drift)
        train_w, test_w = split(cfg, copy.deepcopy(episodes))
        print(f"[rq1] drift_strength={drift}: {len(train_w)} train / {len(test_w)} test windows")
        for pname in POLICY_NAMES:
            policy = policy_factory(pname)
            result = run_experiment_cell(
                train_w,
                test_w,
                CountBaselineDetector(),
                policy,
                detector_name="count_baseline",
                scenario=f"drift_strength={drift}",
                budgets=BUDGETS,
            )
            row = result.to_row()
            row["drift_strength"] = drift
            rows.append(row)
            print(
                f"[rq1]   {pname:16s} AP={result.weighted_ap:.4f} "
                f"Rec@10={result.rec_at_budget[10]:.3f} "
                f"hit_rate={result.episode_hit_rate:.2f}"
            )

    # Spot-check with the torch detector at the strongest drift level only.
    strong_cfg = base_cfg(DRIFT_LEVELS[-1])
    train_w, test_w = split(strong_cfg, copy.deepcopy(episodes))
    for pname in ("frozen", "gated"):
        result = run_experiment_cell(
            train_w,
            test_w,
            FeatureMLPDetector(epochs_per_fit=2, hidden_dim=12),
            policy_factory(pname),
            detector_name="feature_mlp",
            scenario=f"drift_strength={DRIFT_LEVELS[-1]}",
            budgets=BUDGETS,
        )
        row = result.to_row()
        row["drift_strength"] = DRIFT_LEVELS[-1]
        rows.append(row)
        print(f"[rq1] [feature_mlp spot-check] {pname:16s} AP={result.weighted_ap:.4f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "rq1_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[rq1] wrote {csv_path}")

    make_figure(df, out_dir / "rq1_drift_cost.png")


def make_figure(df: pd.DataFrame, path: Path) -> None:
    sub = df[df["detector"] == "count_baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for pname in POLICY_NAMES:
        s = sub[sub["policy"] == pname].sort_values("drift_strength")
        axes[0].plot(s["drift_strength"], s["weighted_ap"], marker="o", label=pname)
        axes[1].plot(s["drift_strength"], s["rec_at_10"], marker="o", label=pname)

    axes[0].set_xlabel("benign drift strength")
    axes[0].set_ylabel("weighted AP")
    axes[0].set_title("RQ1: average precision vs. drift")
    axes[1].set_xlabel("benign drift strength")
    axes[1].set_ylabel("Rec@10 (per-window alert budget)")
    axes[1].set_title("RQ1: recall@budget vs. drift")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[rq1] wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("results/rq1"))
    args = ap.parse_args()
    main(args.out_dir)
