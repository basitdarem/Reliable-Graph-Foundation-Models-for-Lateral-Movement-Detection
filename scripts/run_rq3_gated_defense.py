#!/usr/bin/env python3
"""
RQ3 -- Gated defense: does the proposed `GatedPolicy` recover RQ1's
benign-drift adaptation benefit *without* re-opening RQ2's update-
poisoning gap, at the same fixed alert budget, when both pressures are
present at the same time?

Experiment design
-------------------
A 2x2 design, crossing:

    drift_strength in {NONE, STRONG}    -- RQ1's axis
    conditioning   in {off, on}         -- RQ2's axis

on the same benign background and the same target episode (the RQ2
scenario, reused verbatim via `build_rq2_scenario`-equivalent logic),
for every policy. Two outcomes are reported per cell:

    aggregate_ap, aggregate_rec_at_10   -- whole-stream detection
                                            quality (the RQ1 lens: does
                                            this policy keep up with
                                            drift at all, including on
                                            the *distractor* episodes
                                            that have nothing to do
                                            with the conditioning
                                            attack?)
    hop_rank_frac                       -- the conditioned hop's own
                                            anomaly rank (the RQ2 lens:
                                            did conditioning fool this
                                            policy on the attack it was
                                            built for?)

A policy that only ever looked good on RQ1 (periodic/sliding/drift-
triggered) should show a good `aggregate_rec_at_10` under drift but a
high `hop_rank_frac` jump under conditioning. `FrozenPolicy` should show
the opposite trade: immune to conditioning (`hop_rank_frac` never
moves) but the worst `aggregate_rec_at_10` under strong drift. RQ3's
claim is that `GatedPolicy` should land close to the best adaptive
policy on `aggregate_rec_at_10` under drift, while staying close to
Frozen's near-zero `hop_rank_frac` shift under conditioning -- i.e. it
should not have to choose.

Usage
------
    python scripts/run_rq3_gated_defense.py [--out-dir results/rq3]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lmsafe import attack, synth
from lmsafe.config import LANL_TRAIN_CUTOFF_S, SynthConfig
from lmsafe.data.windows import iter_windows
from lmsafe.detectors import CountBaselineDetector
from lmsafe.drift import PageHinkley
from lmsafe.metrics import WindowScores, recall_at_budget, weighted_ap
from lmsafe.policies import (
    DriftTriggeredPolicy,
    FrozenPolicy,
    GatedPolicy,
    PeriodicPolicy,
    SlidingReplayPolicy,
)

N_BENIGN_DAYS = 30
N_DISTRACTOR_EPISODES = 3
DRIFT_START_DAY = 6
DRIFT_LEVELS = {"none": 0.0, "strong": 0.8}
CONDITIONING_DAYS = 10
CONDITIONING_EVENTS_PER_DAY = 2.0
SEED = 777
TARGET_BUDGET = 10


def policy_factory(name: str):
    if name == "frozen":
        return FrozenPolicy()
    if name == "periodic":
        return PeriodicPolicy(period=24)
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


def build_scenario(drift_strength: float):
    cfg = SynthConfig(
        n_users=40,
        n_hosts=25,
        n_benign_days=N_BENIGN_DAYS,
        base_edges_per_window=25,
        n_episodes=N_DISTRACTOR_EPISODES,
        episode_chain_len=5,
        drift_start_day=DRIFT_START_DAY,
        drift_strength=drift_strength,
        seed=SEED,
    )
    rng = np.random.default_rng(cfg.seed)
    pop = synth.Population(cfg, rng)
    distractor_episodes = synth.default_episodes(cfg, pop, rng)

    target_start_day = N_BENIGN_DAYS - 4
    target = synth.LMEpisode(
        start_s=int(target_start_day * 86400),
        actor_user="U_target@DOM1",
        actor_host="C_target_actor",
        target_hosts=["C_target_1", "C_target_2", "C_target_3", "C_target_4"],
        hop_seconds=30,
    )
    all_episodes = distractor_episodes + [target]
    base_events = synth.generate(cfg, episodes=all_episodes)

    spec = attack.boiling_frog_scenario(
        target,
        conditioning_days=CONDITIONING_DAYS,
        events_per_day=CONDITIONING_EVENTS_PER_DAY,
        seed=SEED,
    )
    conditioned_events = attack.inject_conditioning(base_events, spec)
    return cfg, base_events, conditioned_events, target


def to_windows(events, cfg):
    windows = list(iter_windows(events, cfg.window_seconds))
    train_w = [w for w in windows if w.window_start < LANL_TRAIN_CUTOFF_S]
    test_w = [w for w in windows if w.window_start >= LANL_TRAIN_CUTOFF_S]
    return train_w, test_w


def run_cell(train_w, test_w, policy, target: synth.LMEpisode) -> dict:
    """Run one (detector, policy) over one scenario's test stream and
    report both the RQ1-lens aggregate metrics and the RQ2-lens
    conditioned-hop metric in a single pass (one call to
    `detector.score` per window, as `lmsafe.runner` also guarantees)."""
    detector = CountBaselineDetector()
    policy.fit_initial(detector, train_w)

    actor_host, first_hop_dst = target.actor_host, target.target_hosts[0]
    hop_score: Optional[float] = None
    hop_rank_frac: Optional[float] = None

    records = []
    for w in test_w:
        s = detector.score(w)
        records.append(WindowScores.from_window(w, s))
        if w.window_start <= target.start_s < w.window_start + w.window_seconds:
            for idx, (a, b) in enumerate(w.edges):
                if w.nodes[a] == actor_host and w.nodes[b] == first_hop_dst:
                    order = np.argsort(-s)
                    rank = int(np.where(order == idx)[0][0])
                    hop_score = float(s[idx])
                    hop_rank_frac = rank / max(len(s) - 1, 1)
                    break
        detector = policy.observe(w, s, detector)

    return {
        "aggregate_ap": weighted_ap(records),
        "aggregate_rec_at_10": recall_at_budget(records, TARGET_BUDGET),
        "hop_score": hop_score,
        "hop_rank_frac": hop_rank_frac,
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for drift_label, drift_strength in DRIFT_LEVELS.items():
        cfg, base_events, conditioned_events, target = build_scenario(drift_strength)
        for conditioning_label, events in (("off", base_events), ("on", conditioned_events)):
            train_w, test_w = to_windows(events, cfg)
            for pname in POLICY_NAMES:
                policy = policy_factory(pname)
                m = run_cell(train_w, test_w, policy, target)
                rows.append(
                    {
                        "drift": drift_label,
                        "conditioning": conditioning_label,
                        "policy": pname,
                        **m,
                    }
                )
                print(
                    f"[rq3] drift={drift_label:6s} conditioning={conditioning_label:3s} "
                    f"{pname:16s} AP={m['aggregate_ap']:.4f} Rec@10={m['aggregate_rec_at_10']:.3f} "
                    f"hop_rank_frac={m['hop_rank_frac']:.3f}"
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "rq3_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[rq3] wrote {csv_path}")
    make_figure(df, out_dir / "rq3_gated_defense.png")


def make_figure(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1 (RQ1 lens): Rec@10 under strong drift, conditioning off.
    sub = df[(df.drift == "strong") & (df.conditioning == "off")].set_index("policy").reindex(POLICY_NAMES)
    axes[0].bar(sub.index, sub["aggregate_rec_at_10"])
    axes[0].set_title("RQ1 lens: Rec@10 under strong drift\n(no conditioning)")
    axes[0].set_ylabel("Rec@10")
    axes[0].tick_params(axis="x", rotation=20)

    # Panel 2 (RQ2 lens): hop rank shift (conditioning on - off), drift=none.
    off = df[(df.drift == "none") & (df.conditioning == "off")].set_index("policy")["hop_rank_frac"]
    on = df[(df.drift == "none") & (df.conditioning == "on")].set_index("policy")["hop_rank_frac"]
    shift = (on - off).reindex(POLICY_NAMES)
    axes[1].bar(shift.index, shift.values, color="darkorange")
    axes[1].set_title("RQ2 lens: conditioned-hop rank shift\n(higher = more fooled)")
    axes[1].set_ylabel("hop_rank_frac(on) - hop_rank_frac(off)")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("RQ3: can GatedPolicy win on both axes at once?")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[rq3] wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("results/rq3"))
    args = ap.parse_args()
    main(args.out_dir)
