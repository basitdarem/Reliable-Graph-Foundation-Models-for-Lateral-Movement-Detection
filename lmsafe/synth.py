"""
Synthetic authentication-event generator with known ground truth.

Why a synthetic generator belongs in a paper about LANL/OpTC
---------------------------------------------------------------
Every module downstream of `lmsafe.data` (windowing, detectors, drift
detection, update policies, the attacker, metrics) needs to be correct
*before* it is pointed at a 7.2 GB stream that takes an hour to read. This
module produces a small, fast, fully-controlled event stream in the same
`Event` shape LANL/OpTC produce, with two properties real data cannot give
a unit test:

    1. Exact, known ground truth for both the benign drift and the
       injected lateral-movement episodes (so metrics.py can be tested
       against a known-correct answer, not just "a plausible-looking
       number").
    2. A dial (`SynthConfig.drift_strength`) to turn benign drift on or
       off independently of the attack, which is exactly the
       counterfactual RQ1 asks for and which LANL alone cannot offer
       (LANL's drift and its one real intrusion are both fixed, whatever
       they happen to be in that one 58-day recording).

Generative model (deliberately simple, documented limitations below)
-----------------------------------------------------------------------
- A population of users and hosts exists from day 0. Each user has one
  "home" host (their desktop). A small subset of hosts are "common
  servers" (file/domain-controller-style) that most users talk to.
- Each benign event picks a user, then a destination: with high
  probability a common server (Zipf-weighted, matching the real LANL
  red-team inspection showing a small number of very popular destination
  hosts -- see README), otherwise another user's home host (modeling
  occasional legitimate peer-to-peer / admin access).
- Starting at `drift_start_day`, the host and user population grows
  monotonically (new hosts/users are added, none are ever removed). This
  is "collective, persistent" drift by construction: it shifts the
  *whole* population's traffic mix, gradually and permanently, which is
  exactly the RQ1 hypothesis's definition of benign drift as opposed to
  an attacker's local, narrow conditioning (see `lmsafe.attack`).
- `n_episodes` lateral-movement episodes are injected at well-separated
  times: one actor host authenticates, as one user, to a rapid chain of
  `episode_chain_len` distinct target hosts a few seconds/minutes apart --
  the same "one host touches many others quickly" signature observed in
  the real LANL red-team file (701 of 749 real events originate from a
  single host, C17693).

Known limitations (state them, don't hide them)
---------------------------------------------------
- No user/host churn (nothing is ever removed) -- real organizations lose
  machines and accounts, not just gain them. Acceptable for this
  project's purpose (testing the pipeline, not modeling an enterprise),
  but should not be oversold as realistic in the paper.
- All events are already in the "kept" shape the fair LANL protocol
  selects for (orientation="LogOn", src_comp != dst_comp): the generator
  tests everything *after* extraction, not the extraction/filtering
  logic itself (that is tested against the real LANL schema in
  tests/test_lanl_labels.py).
- Auth-type/success fields are set but not modeled adversarially; they
  are not used by any detector or policy in this project (the fair
  protocol explicitly avoids filtering on auth_type -- see
  lmsafe.config.LanlConfig.filter_ntlm_only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SynthConfig
from .data.schema import Event

COMMON_SERVER_FRACTION = 0.1  # fraction of the initial host pool that are shared servers


@dataclass(frozen=True)
class LMEpisode:
    """One injected lateral-movement episode, with known ground truth.

    Attributes
    ----------
    start_s : int
        Time of the first hop.
    actor_user : str
        The (compromised) user identity used for every hop in this
        episode.
    actor_host : str
        The single source host every hop originates from -- mirrors the
        real LANL red team's heavy concentration on one source host.
    target_hosts : list[str]
        Distinct destination hosts visited, in order, one per hop.
    hop_seconds : int
        Spacing between consecutive hops.
    """

    start_s: int
    actor_user: str
    actor_host: str
    target_hosts: list[str]
    hop_seconds: int = 30

    def events(self) -> list[Event]:
        out = []
        for i, dst in enumerate(self.target_hosts):
            out.append(
                Event(
                    t=self.start_s + i * self.hop_seconds,
                    src_user=self.actor_user,
                    dst_user=self.actor_user,
                    src_comp=self.actor_host,
                    dst_comp=dst,
                    auth_type="NTLM",
                    logon_type="LogOn",
                    orientation="LogOn",
                    success=True,
                    is_malicious=True,
                    source="synthetic",
                )
            )
        return out


class Population:
    """Tracks the growing user/host population as a function of elapsed
    simulated days, implementing the "collective, persistent drift"
    model described in the module docstring.

    Public (not an internal detail) specifically so an experiment
    script can build one population once, place a fixed set of episodes
    on it with `default_episodes`, and then re-run `generate` with
    those *same* episodes across several `drift_strength` values for a
    clean counterfactual comparison -- exactly what
    `scripts/run_rq1_drift_cost.py` does. `generate()` itself builds and
    discards its own `Population` internally when called without
    pre-built `episodes`.
    """

    def __init__(self, cfg: SynthConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        n_common = max(1, int(round(cfg.n_hosts * COMMON_SERVER_FRACTION)))
        self.common_servers: list[str] = [f"SRV{i}" for i in range(n_common)]
        # Zipf-like popularity weights among common servers (a few are
        # much more popular than the rest), matching the skew observed
        # in the real LANL destination-host counts.
        ranks = np.arange(1, n_common + 1)
        self.common_weights = 1.0 / ranks
        self.common_weights /= self.common_weights.sum()

        self.hosts: list[str] = [f"C{i}" for i in range(cfg.n_hosts - n_common)]
        self.users: list[str] = [f"U{i}@DOM1" for i in range(cfg.n_users)]
        self.home_host: dict[str, str] = {
            u: self.hosts[i % len(self.hosts)] for i, u in enumerate(self.users)
        }
        self._next_host_id = len(self.hosts)
        self._next_user_id = len(self.users)

    def grow(self, day: int) -> None:
        """Apply one day's worth of drift-driven population growth."""
        cfg = self.cfg
        if cfg.drift_start_day is None or day < cfg.drift_start_day:
            return
        if cfg.drift_strength <= 0:
            return
        new_hosts = max(0, int(round(cfg.drift_strength * 0.05 * cfg.n_hosts)))
        new_users = max(0, int(round(cfg.drift_strength * 0.05 * cfg.n_users)))
        for _ in range(new_hosts):
            h = f"C{self._next_host_id}"
            self._next_host_id += 1
            self.hosts.append(h)
        for _ in range(new_users):
            u = f"U{self._next_user_id}@DOM1"
            self._next_user_id += 1
            self.users.append(u)
            self.home_host[u] = self.rng.choice(self.hosts)

    def sample_event(self, t: int) -> Event:
        # numpy's Generator.choice on a list of Python strings returns
        # numpy.str_ scalars; cast back to plain str so every downstream
        # consumer (dict keys, CSV writers, dataclass reprs) sees exactly
        # the same type real LANL/OpTC readers produce.
        u = str(self.rng.choice(self.users))
        src = self.home_host[u]
        if self.rng.random() < 0.7:
            dst = str(self.rng.choice(self.common_servers, p=self.common_weights))
        else:
            peer = str(self.rng.choice(self.users))
            dst = self.home_host[peer]
            if dst == src:
                dst = str(self.rng.choice(self.common_servers, p=self.common_weights))
        return Event(
            t=t,
            src_user=u,
            dst_user=u,
            src_comp=src,
            dst_comp=dst,
            auth_type="Kerberos" if self.rng.random() < 0.85 else "NTLM",
            logon_type="LogOn",
            orientation="LogOn",
            success=True,
            is_malicious=False,
            source="synthetic",
        )


def default_episodes(
    cfg: SynthConfig,
    population: Population,
    rng: np.random.Generator,
    min_start_s: int = 2 * 86400,
) -> list[LMEpisode]:
    """Place `cfg.n_episodes` lateral-movement episodes at evenly-spaced,
    randomly-jittered times across the benign period, each using a fresh
    actor host/user and a chain of distinct, randomly chosen targets.

    `min_start_s` keeps every episode safely clear of the benign
    training cutoff used throughout this project
    (`lmsafe.config.LANL_TRAIN_CUTOFF_S`, 41 hours): with a short
    `n_benign_days` and several episodes, an un-clamped evenly-spaced
    placement can land the first episode inside what `split_train_test`
    treats as the training window, silently removing it from the test
    stream's attack population rather than erroring. Defaulting to 2
    days gives a comfortable margin past 41 hours without needing the
    caller to know that constant.
    """
    total_s = cfg.n_benign_days * 86400
    if cfg.n_episodes <= 0:
        return []
    usable_s = max(total_s - min_start_s, 1)
    slot = usable_s / cfg.n_episodes
    episodes = []
    for i in range(cfg.n_episodes):
        jitter = rng.uniform(0.2, 0.8)
        start_s = int(min_start_s + i * slot + jitter * slot)
        actor_user = str(rng.choice(population.users))
        actor_host = population.home_host[actor_user]
        n_targets = cfg.episode_chain_len
        pool = [h for h in population.hosts if h != actor_host] + population.common_servers
        targets = [str(h) for h in rng.choice(pool, size=min(n_targets, len(pool)), replace=False)]
        episodes.append(
            LMEpisode(
                start_s=start_s,
                actor_user=actor_user,
                actor_host=actor_host,
                target_hosts=targets,
                hop_seconds=cfg.episode_hop_seconds,
            )
        )
    return episodes


def generate(
    cfg: SynthConfig, episodes: list[LMEpisode] | None = None
) -> list[Event]:
    """Generate a full, time-sorted synthetic event stream.

    Parameters
    ----------
    cfg : SynthConfig
    episodes : list[LMEpisode], optional
        Lateral-movement episodes to merge in. If omitted, `n_episodes`
        default episodes are placed automatically via `default_episodes`.
        Pass `episodes=[]` explicitly to get a purely benign stream (used
        by RQ1 experiments that measure drift cost in isolation, with no
        attack at all).

    Returns
    -------
    list[Event]
        Sorted ascending by `.t`. Materialized (not a generator) because
        the synthetic datasets used in this project are small enough
        (tens of thousands of events) that eager generation makes the
        rest of the pipeline simpler to reason about and test.
    """
    rng = np.random.default_rng(cfg.seed)
    pop = Population(cfg, rng)

    if episodes is None:
        episodes = default_episodes(cfg, pop, rng)

    events: list[Event] = []
    n_days = cfg.n_benign_days
    for day in range(n_days):
        pop.grow(day)
        day_start = day * 86400
        drift_mult = 1.0
        if cfg.drift_start_day is not None and day >= cfg.drift_start_day:
            drift_mult = 1.0 + cfg.drift_strength  # more traffic as the org grows
        n_windows = 86400 // cfg.window_seconds
        for w in range(n_windows):
            window_start = day_start + w * cfg.window_seconds
            n_events = rng.poisson(cfg.base_edges_per_window * drift_mult)
            for _ in range(n_events):
                t = int(window_start + rng.integers(0, cfg.window_seconds))
                events.append(pop.sample_event(t))

    for ep in episodes:
        events.extend(ep.events())

    events.sort(key=lambda e: e.t)
    return events
