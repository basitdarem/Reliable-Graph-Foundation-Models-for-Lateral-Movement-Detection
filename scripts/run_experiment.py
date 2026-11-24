#!/usr/bin/env python3
"""
Generic, config-driven experiment runner.

This is the one script meant to be pointed at the *real* LANL (or,
once `lmsafe.data.optc` is implemented, OpTC) data on Colab -- every
other script under `scripts/` is a self-contained synthetic
demonstration of one research question. A YAML config says which
dataset to use, which detector, and which policies to compare; this
script builds the data once, runs every policy against it, and writes
one results CSV.

Config schema
---------------
    source: synthetic | lanl          # which lmsafe.data reader to use

    detector:
      name: count_baseline | feature_mlp
      kwargs: {...}                    # forwarded to the detector class

    policies:
      - name: frozen | periodic | sliding_replay | drift_triggered | gated
        kwargs: {...}                  # forwarded to the policy class
                                        # (drift_triggered additionally
                                        # takes drift_detector_name:
                                        # page_hinkley|adwin and
                                        # drift_detector_kwargs)

    budgets: [1, 5, 10, 20]            # per-window alert budgets for Rec@B

    output:
      dir: results/my_run
      csv: results.csv

    # when source: synthetic -- see lmsafe.config.SynthConfig fields
    synth:
      n_users: 40
      ...

    # when source: lanl -- see lmsafe.config.LanlConfig fields
    lanl:
      end_s: 2592000                   # LANL_DEV_CUTOFF_S (30 days); use
                                        # LANL_FULL_CUTOFF_S (58 days) for
                                        # final paper numbers
      train_cutoff_s: 147600           # LANL_TRAIN_CUTOFF_S (41 hours)
      only_remote_logon: true
      filter_ntlm_only: false
      cache_path: lmsafe_cache/lanl_dev_windows.pkl
                                        # OPTIONAL but strongly recommended:
                                        # the streamed-and-windowed
                                        # (train, test) WindowGraph lists
                                        # are pickled here on first run and
                                        # loaded directly on every later
                                        # run, so a second detector/policy
                                        # sweep never re-streams the 7.2 GB
                                        # auth file (and keeps working even
                                        # after the signed LANL link has
                                        # expired). Written incrementally
                                        # (see checkpoint_every below), not
                                        # only once at the end, so an
                                        # interrupted multi-hour run still
                                        # leaves a usable partial file.
      checkpoint_every: 24              # OPTIONAL, only used with cache_path:
                                        # write a partial checkpoint every N
                                        # test windows (default 24, roughly
                                        # once per simulated day at the
                                        # default 1-hour window length).

Example configs are in `configs/`.

Usage
------
    python scripts/run_experiment.py configs/synthetic_quick.yaml
    python scripts/run_experiment.py configs/lanl_dev.yaml       # on Colab,
                                                                    # with
                                                                    # LMSAFE_LANL_BASE_URL set
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import yaml

from lmsafe.config import LanlConfig, SynthConfig
from lmsafe.data import lanl
from lmsafe.data.windows import WindowGraph, iter_windows
from lmsafe.factories import make_detector, make_policy
from lmsafe.runner import run_experiment_cell


def build_synthetic_windows(cfg_dict: dict) -> tuple[list[WindowGraph], list[WindowGraph]]:
    from lmsafe import synth
    from lmsafe.config import LANL_TRAIN_CUTOFF_S

    # The training cutoff is always lmsafe.config.LANL_TRAIN_CUTOFF_S
    # (41 hours), reused here for synthetic data too so every scenario
    # in this project shares exactly one definition of "the benign
    # training period" -- see that constant's own docstring.
    cfg = SynthConfig(**cfg_dict)
    events = synth.generate(cfg)
    windows = list(iter_windows(events, cfg.window_seconds))
    train_w = [w for w in windows if w.window_start < LANL_TRAIN_CUTOFF_S]
    test_w = [w for w in windows if w.window_start >= LANL_TRAIN_CUTOFF_S]
    return train_w, test_w


def build_lanl_windows(cfg_dict: dict) -> tuple[list[WindowGraph], list[WindowGraph]]:
    cache_path = cfg_dict.pop("cache_path", None)
    # How often (in test windows) to write a crash-safe partial
    # checkpoint to `cache_path` while streaming -- see the loop below.
    # Only meaningful when `cache_path` is set, since there is nowhere
    # else to checkpoint to.
    checkpoint_every = cfg_dict.pop("checkpoint_every", 24)
    if cache_path and Path(cache_path).exists():
        print(f"[run_experiment] loading cached LANL windows from {cache_path}")
        with open(cache_path, "rb") as fh:
            cached_train_w, cached_test_w = pickle.load(fh)
        # This cache may be a *finished* run's output or an interrupted
        # run's partial checkpoint -- the pickle format is identical
        # either way (see the long comment below), so this is the only
        # place that can warn a human the loaded data might be short of
        # the config's own `end_s`. Confirmed necessary in practice: a
        # real 58-day run was Ctrl+C'd at day 40.64 after repeated
        # network reconnects, and silently reloading that partial cache
        # on a later run would otherwise look identical to a genuine,
        # complete 58-day run in every later log line.
        requested_days = cfg_dict.get("end_s", 0) / 86400
        if cached_test_w:
            covered_days = max(w.window_start + w.window_seconds for w in cached_test_w) / 86400
        else:
            covered_days = 0.0
        print(
            f"[run_experiment] cache contains {len(cached_train_w)} train windows + "
            f"{len(cached_test_w)} test windows, covering up to day {covered_days:.2f} "
            f"of the {requested_days:.1f} days this config asks for"
        )
        if requested_days and covered_days < requested_days - 0.05:
            print(
                f"[run_experiment] WARNING: this cache covers only {covered_days:.2f} of "
                f"{requested_days:.1f} requested days -- it looks like a partial checkpoint "
                f"from an interrupted run, not a finished one. Metrics computed from it are "
                f"real but describe a SHORTER evaluation window than this config's label "
                f"implies. To get the full window, delete this cache file and rerun (the "
                f"network stream itself cannot resume mid-file, so this re-reads from the "
                f"start); to knowingly keep the partial window, that is fine too -- just "
                f"don't report it as the full-window number."
            )
        return cached_train_w, cached_test_w

    cfg = LanlConfig(**{k: v for k, v in cfg_dict.items() if k != "cache_dir"})
    print(
        f"[run_experiment] streaming LANL auth events up to t={cfg.end_s:,}s "
        f"({cfg.end_s / 86400:.1f} days) -- this reads from the network, "
        "see lmsafe.data.lanl_stream for the signed-link setup"
    )
    base_url = lanl.lanl_stream.resolve_base_url()
    redteam = lanl.load_redteam_records(base_url)
    events_iter = lanl.stream_auth_events(cfg, base_url=base_url, redteam=redteam)
    train_events, test_events_iter = lanl.split_train_test(events_iter, cfg)

    train_w = list(iter_windows(train_events, cfg.window_seconds))

    # Stream-build the test-window list with periodic crash-safe
    # checkpoints to `cache_path`, instead of only writing it once at
    # the very end. A real LANL run (58 days = 1B+ raw lines) can take
    # hours, and a Colab disconnect, a Ctrl+C, or a network hiccup
    # before the final line used to leave *nothing* on disk at all --
    # the exact failure reported: a 58-day run reached day ~50 of 58,
    # was interrupted, and Google Drive came up `total 0` because the
    # single pickle.dump() at the end of this function never ran. Each
    # checkpoint is a complete, directly-loadable `(train_w, test_w)`
    # pickle -- the `if cache_path and Path(cache_path).exists()` branch
    # above cannot tell a checkpoint from a finished run, so a resumed
    # session picks up the most recent checkpoint transparently (at the
    # cost of re-covering the tail that wasn't yet checkpointed, since
    # this does not resume the network stream itself -- only the
    # output).
    test_w: list = []
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        for w in iter_windows(test_events_iter, cfg.window_seconds):
            test_w.append(w)
            if len(test_w) % checkpoint_every == 0:
                lanl.atomic_pickle_dump((train_w, test_w), Path(cache_path))
                print(
                    f"[run_experiment] checkpoint: {len(train_w)} train windows + "
                    f"{len(test_w)} test windows so far -> {cache_path}"
                )
        lanl.atomic_pickle_dump((train_w, test_w), Path(cache_path))
        print(f"[run_experiment] cached windows to {cache_path} for future runs")
    else:
        test_w = list(iter_windows(test_events_iter, cfg.window_seconds))

    return train_w, test_w


def main(config_path: Path) -> None:
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    source = config["source"]
    if source == "synthetic":
        train_w, test_w = build_synthetic_windows(dict(config.get("synth", {})))
    elif source == "lanl":
        train_w, test_w = build_lanl_windows(dict(config.get("lanl", {})))
    else:
        sys.exit(f"unknown source {source!r}, expected 'synthetic' or 'lanl'")

    print(f"[run_experiment] {len(train_w)} train windows, {len(test_w)} test windows")

    budgets = tuple(config.get("budgets", [1, 5, 10, 20]))
    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / config["output"].get("csv", "results.csv")

    det_cfg = config["detector"]
    rows = []
    for pol_cfg in config["policies"]:
        detector = make_detector(det_cfg["name"], **det_cfg.get("kwargs", {}))
        policy = make_policy(pol_cfg["name"], **pol_cfg.get("kwargs", {}))
        result = run_experiment_cell(
            train_w,
            test_w,
            detector,
            policy,
            detector_name=det_cfg["name"],
            scenario=config.get("scenario_label", source),
            budgets=budgets,
        )
        print(
            f"[run_experiment] {policy.name:16s} AP={result.weighted_ap:.4f} "
            f"AUC={result.weighted_auc:.4f} "
            f"hit_rate={result.episode_hit_rate:.3f} "
            f"(n_episodes={result.n_episodes})"
        )
        rows.append(result.to_row())

    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[run_experiment] wrote {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="path to a YAML experiment config")
    args = ap.parse_args()
    main(args.config)
