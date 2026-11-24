#!/usr/bin/env python3
"""
RQ2 -- Update poisoning: how much detection quality do the naive update
policies lose when an adaptive attacker injects conditioning edges
before attacking, compared to the same policies with no conditioning at
all?

Experiment design
-------------------
One benign background (fixed `drift_strength`, fixed seed) with one
"target" lateral-movement episode. Two variants of the event stream are
built from the *same* base events and the *same* episode:

    clean        -- the episode alone, no conditioning.
    conditioned  -- `lmsafe.attack.boiling_frog_scenario` adds a slow,
                     low-volume trickle from the episode's own actor
                     host to its first target host, for
                     `CONDITIONING_DAYS` days immediately before the
                     episode.

Every policy is run on both variants; the quantity of interest is the
*drop* in detection quality on the target episode specifically (not the
aggregate AUC/AP across the whole stream, which is dominated by the
other, unconditioned episodes also present as distractors) between
`clean` and `conditioned`. A policy robust to this attack should show
little to no drop; a policy that blindly incorporates everything it
sees should show a large one.

`GatedPolicy` is included in this comparison deliberately: RQ2 is where
its score-gated admission and multi-host-corroboration mechanisms
either do or do not earn their keep against the other four baselines
run identically. RQ3's script then asks the complementary question (can
Gated also recover RQ1's benign-drift performance at the same time).

Usage
------
    python scripts/run_rq2_poisoning.py [--out-dir results/rq2]
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
from lmsafe.policies import (
    DriftTriggeredPolicy,
    FrozenPolicy,
    GatedPolicy,
    PeriodicPolicy,
    SlidingReplayPolicy,
)
from lmsafe.drift import PageHinkley

N_BENIGN_DAYS = 30
N_DISTRACTOR_EPISODES = 3     # other, unconditioned episodes present as realistic background
DRIFT_STRENGTH = 0.2
DRIFT_START_DAY = 6
CONDITIONING_DAYS = 10
CONDITIONING_EVENTS_PER_DAY = 2.0
SEED = 123
TARGET_BUDGET = 10  # per-window alert budget used to score the target episode specifically


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


def build_scenario():
    cfg = SynthConfig(
        n_users=40,
        n_hosts=25,
        n_benign_days=N_BENIGN_DAYS,
        base_edges_per_window=25,
        n_episodes=N_DISTRACTOR_EPISODES,
        episode_chain_len=5,
        drift_start_day=DRIFT_START_DAY,
        drift_strength=DRIFT_STRENGTH,
        seed=SEED,
    )
    rng = np.random.default_rng(cfg.seed)
    pop = synth.Population(cfg, rng)
    distractor_episodes = synth.default_episodes(cfg, pop, rng)

    # The RQ2 *target* episode: placed late enough that a full
    # CONDITIONING_DAYS campaign still starts safely after the 41-hour
    # training cutoff, using a fresh actor not reused by any distractor.
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


def target_hop_metrics(test_w, detector, policy, target: synth.LMEpisode) -> dict:
    """Run one (detector, policy) over `test_w` and report how
    suspicious the detector finds *specifically* the conditioned hop --
    `target.actor_host -> target.target_hosts[0]` -- in the window the
    real attack happens.

    This -- not "was the whole episode ever caught" -- is the right
    RQ2 measurement: `boiling_frog_scenario` only conditions the first
    hop's destination. Hops 2+ (`target_hosts[1:]`) are, by
    construction, hosts the actor has never touched before regardless
    of conditioning, so they are almost always individually flagged as
    novel and would make an episode-level "caught anywhere" check
    insensitive to the attack this experiment is actually studying.
    Comparing the conditioned hop's own score/rank between the `clean`
    and `conditioned` variants isolates exactly the effect RQ2 asks
    about.

    `detector`/`policy` must already have had `policy.fit_initial`
    called on them by the caller -- this function only runs the
    test-window loop. Calling `fit_initial` again here, even with an
    empty window list, would silently wipe the real 41-hour benign fit
    (every concrete `Detector.fit(..., warm_start=False)` resets to an
    untrained state first) right before scoring.
    """
    actor_host = target.actor_host
    first_hop_dst = target.target_hosts[0]

    hop_score: Optional[float] = None
    hop_rank_frac: Optional[float] = None
    hop_in_budget: Optional[bool] = None

    for w in test_w:
        s = detector.score(w)
        if w.window_start <= target.start_s < w.window_start + w.window_seconds:
            for idx, (a, b) in enumerate(w.edges):
                if w.nodes[a] == actor_host and w.nodes[b] == first_hop_dst:
                    order = np.argsort(-s)
                    rank = int(np.where(order == idx)[0][0])
                    hop_score = float(s[idx])
                    hop_rank_frac = rank / max(len(s) - 1, 1)
                    hop_in_budget = rank < TARGET_BUDGET
                    break
        detector = policy.observe(w, s, detector)

    return {
        "hop_score": hop_score,
        "hop_rank_frac": hop_rank_frac,
        "hop_in_budget": hop_in_budget,
    }


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, base_events, conditioned_events, target = build_scenario()
    print(f"[rq2] target episode at day {target.start_s / 86400:.1f}, "
          f"actor={target.actor_host}, first_target={target.target_hosts[0]}")

    rows = []
    for variant_name, events in (("clean", base_events), ("conditioned", conditioned_events)):
        train_w, test_w = to_windows(events, cfg)
        for pname in POLICY_NAMES:
            detector = CountBaselineDetector()
            policy = policy_factory(pname)
            policy.fit_initial(detector, train_w)
            m = target_hop_metrics(test_w, detector, policy, target)
            rows.append({"variant": variant_name, "policy": pname, **m})
            print(
                f"[rq2] {variant_name:11s} {pname:16s} "
                f"score={m['hop_score']:.4f} rank_frac={m['hop_rank_frac']:.4f} "
                f"in_budget={m['hop_in_budget']}"
            )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "rq2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[rq2] wrote {csv_path}")
    make_figure(df, out_dir / "rq2_poisoning.png")


def make_figure(df: pd.DataFrame, path: Path) -> None:
    piv = df.pivot(index="policy", columns="variant", values="hop_rank_frac").reindex(POLICY_NAMES)
    piv = piv.astype(float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(piv.index))
    width = 0.35
    ax.bar(x - width / 2, piv["clean"], width, label="clean (no conditioning)")
    ax.bar(x + width / 2, piv["conditioned"], width, label="conditioned (boiling-frog)")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.set_ylabel("conditioned hop's anomaly rank (fraction, 0=most suspicious)")
    ax.set_title("RQ2: effect of update poisoning on the conditioned hop's rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[rq2] wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("results/rq2"))
    args = ap.parse_args()
    main(args.out_dir)
