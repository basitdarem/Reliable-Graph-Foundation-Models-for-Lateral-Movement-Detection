"""
Central configuration constants and small config dataclasses.

Every *time* constant in this file is in seconds since the start of the
LANL "cyber1" dataset (its own internal clock, not a wall-clock date -- the
released dataset was de-identified and re-timestamped by LANL). Getting
these numbers right is the single most important correctness property of
the whole pipeline: a wrong split silently turns an "unseen test attack"
into "an attack the model trained on", which inflates every downstream
metric without raising any error.

References for the constants below
------------------------------------
- Kent, A. D. (2015). *Comprehensive, Multi-Source Cyber-Security Events*.
  Los Alamos National Laboratory. https://doi.org/10.17021/1179829
- Larroche, T. (2026). LMDEval fair-evaluation protocol for lateral-movement
  detection (BSD-2-Clause). The temporal split and "keep all remote LogOn
  events, don't filter to NTLM-only" rules are adopted directly from this
  protocol so our numbers are comparable to it. See NOTICE.md.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# LANL "cyber1" temporal constants (seconds since the dataset's own epoch)
# ---------------------------------------------------------------------------

#: One hour / one day in seconds, spelled out because every other constant
#: below is defined as a multiple of one of these -- never hardcode a second
#: copy of "86400" elsewhere, import SECONDS_PER_DAY instead.
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

#: The first red-team (ground-truth malicious) event in the public LANL
#: redteam.txt.gz is at t = 150885 s = 41.9h. Training on anything at or
#: after this point risks leaking attack traffic into the "benign" model.
#: We cut training at a round 41 hours, strictly before this.
LANL_REDTEAM_FIRST_EVENT_S = 150_885
LANL_TRAIN_CUTOFF_S = 41 * SECONDS_PER_HOUR  # 147,600 s -- matches LMDEval

#: The full released auth log spans 58 days, but (as of this project's own
#: inspection of redteam.txt.gz on 2026-11-13) every one of the 749 labeled
#: red-team events falls before day 29.6 and there are zero events after
#: day 30. We therefore default every *development* run to a 30-day cut
#: (LANL_DEV_CUTOFF_S) -- this already contains 100% of the known attack
#: episodes while reading roughly half the 7.2 GB auth stream. Final paper
#: numbers should additionally be reported on the full LANL_FULL_CUTOFF_S
#: window to match LMDEval exactly; set --end-s accordingly in the scripts.
LANL_DEV_CUTOFF_S = 30 * SECONDS_PER_DAY        # 2,592,000 s
LANL_FULL_CUTOFF_S = 58 * SECONDS_PER_DAY       # 5,011,200 s

#: LMDEval's own ablation window (14 days) -- useful for a quick, still
#: fairly complete, run: it contains 628 / 749 (84%) of the red-team events.
LANL_QUICK_CUTOFF_S = 14 * SECONDS_PER_DAY      # 1,209,600 s

#: OpTC (second dataset, used for the transfer / generalization check in a
#: later chapter, not for RQ1-RQ3): 6 days train, last 3 days test, per the
#: fair protocol in the project summary.
OPTC_TRAIN_DAYS = 6
OPTC_TEST_LAST_DAYS = 3

#: Window size used to turn the raw event stream into a sequence of graphs.
#: One hour balances graph density (enough edges per window to fit a model)
#: against responsiveness (how quickly a policy can react to drift/attack).
DEFAULT_WINDOW_SECONDS = SECONDS_PER_HOUR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Default cache directory for anything downloaded/extracted once and
#: reused across runs (the aggregated LANL window-graphs, OpTC extracts).
#: Override via the LMSAFE_CACHE_DIR environment variable, e.g. to point at
#: a mounted Google Drive folder on Colab.
import os  # noqa: E402

DEFAULT_CACHE_DIR = Path(os.environ.get("LMSAFE_CACHE_DIR", "./lmsafe_cache"))


@dataclasses.dataclass(frozen=True)
class LanlConfig:
    """Configuration for one LANL extraction / experiment run.

    Notes
    -----
    `base_url` is intentionally *not* a field with a default value, and
    it is never read from a config file on disk: it is a short-lived
    (~8 hour) signed credential handed out by
    https://csr.lanl.gov/data/cyber1/ after you register and accept the
    data-use terms there. See `lmsafe.data.lanl_stream.resolve_base_url`
    for how it is obtained at run time instead of being hardcoded.
    """

    end_s: int = LANL_DEV_CUTOFF_S
    start_s: int = 0
    train_cutoff_s: int = LANL_TRAIN_CUTOFF_S
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    only_remote_logon: bool = True   # the LMDEval fair-protocol filter
    filter_ntlm_only: bool = False   # keep False -- that filter is the
                                      # non-fair shortcut this project
                                      # deliberately avoids (see README).
    cache_dir: Path = DEFAULT_CACHE_DIR


@dataclasses.dataclass(frozen=True)
class SynthConfig:
    """Configuration for the synthetic event generator (lmsafe.synth).

    The synthetic generator exists to unit-test and smoke-test every
    module *downstream* of data extraction (windows, detectors, drift,
    policies, attack, metrics) without any network access and with a
    dataset small enough to run on a 2-CPU, no-GPU sandbox in seconds.
    It is not a substitute for the real LANL/OpTC results in the paper.
    """

    n_users: int = 60
    n_hosts: int = 40
    n_benign_days: int = 20
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    base_edges_per_window: int = 250
    drift_start_day: Optional[int] = 10     # when benign drift begins
    drift_strength: float = 0.5             # 0 = none, 1 = strong
    n_episodes: int = 6                     # injected lateral-movement episodes
    episode_chain_len: int = 5               # hops per episode (actor -> N targets)
    episode_hop_seconds: int = 30            # spacing between hops in one episode
    seed: int = 0
