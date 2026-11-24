"""
Update-poisoning / "boiling-frog" conditioning injector -- RQ2 ONLY.

Scope and intent
-------------------
This module exists to answer one measurement question inside this
project's own **offline evaluation harness**: *if an adversary can
influence what a continually-updating detector learns is "normal"
before it attacks, how much does that degrade the detector's ability to
flag the real attack when it finally happens, under each update policy
in `lmsafe.policies`?*

It operates exclusively on event streams already materialized in memory
(the synthetic generator's output, or a pre-extracted/labeled LANL
window list) inside `lmsafe.runner`'s experiment loop. It has no network
access, no ability to reach any live system, and no code path that
writes anywhere outside the Python objects passed into it. It is the
offline-research equivalent of a published adversarial-robustness
ablation (the kind already present in, e.g., the `poison_datasets.py`
ablation in the CyberGFM baseline this project reviewed, and in the
ANTIDOTE / SIGL-style poisoning literature) -- a controlled, synthetic
measurement of a defense's robustness, not an attack tool.

The conditioning scenario
----------------------------
Before a lateral-movement episode (`lmsafe.synth.LMEpisode`) happens, the
same actor host sends a slow, low-volume trickle of authentication
events toward the same eventual target over `conditioning_days` days.
Each individual conditioning event is built to look exactly like an
ordinary benign authentication event (same schema, same `auth_type`/
`logon_type` distribution as the synthetic generator's benign traffic,
low volume) -- the entire point of the scenario is that no single
conditioning event is unusual; only the slow, persistent pattern over
many windows is informative, and only in hindsight.

Ground-truth labeling choice (read before using `recall_at_budget` on a
conditioned run): conditioning events are **not** labeled
`is_malicious=True` by default (`ConditioningSpec.label_as_malicious`).
This mirrors how the real LANL red-team file labels only the logged
lateral-movement authentications themselves, not any reconnaissance or
staging activity that might have preceded them. The primary RQ2 metric
this project reports is therefore "recall on the real attack episode,
with vs. without a preceding conditioning campaign" -- *not* "recall on
the conditioning events" (though setting `label_as_malicious=True` is
supported for a secondary analysis of whether the conditioning phase
itself was ever independently detectable, which is a reasonable thing to
also report in the paper if space allows).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.schema import Event
from .synth import LMEpisode


@dataclass(frozen=True)
class ConditioningSpec:
    """Parameters of one boiling-frog conditioning campaign.

    Attributes
    ----------
    start_s, end_s : int
        Conditioning window; by convention `end_s` should be at or
        before the target `LMEpisode.start_s` (see
        `boiling_frog_scenario`, which enforces this automatically).
    actor_user, actor_host : str
        Same identity used for the later real attack -- the scenario
        models one already-compromised foothold being used both to
        condition the model and, later, to actually move laterally.
    target_host : str
        The host the attacker wants the detector to have already
        learned is "a normal destination for actor_host" by the time
        the real attack happens.
    events_per_day : float
        Mean conditioning events per day (Poisson rate). Kept low by
        default specifically so the aggregate per-window statistics a
        `lmsafe.drift` detector watches do not shift sharply -- see
        `lmsafe.policies.DriftTriggeredPolicy`'s docstring.
    label_as_malicious : bool
        See module docstring. Default False.
    seed : int
        RNG seed for the Poisson arrival process and jitter.
    """

    start_s: int
    end_s: int
    actor_user: str
    actor_host: str
    target_host: str
    events_per_day: float = 2.0
    label_as_malicious: bool = False
    seed: int = 0


def generate_conditioning_events(spec: ConditioningSpec) -> list[Event]:
    """Generate the conditioning event stream for one `ConditioningSpec`.

    Arrivals follow a homogeneous Poisson process at rate
    `events_per_day` over `[start_s, end_s]` -- a deliberately
    unremarkable, memoryless arrival pattern (no burstiness that would
    itself be a detectable signature).
    """
    if spec.end_s <= spec.start_s:
        return []
    rng = np.random.default_rng(spec.seed)
    duration_days = (spec.end_s - spec.start_s) / 86400.0
    n_events = rng.poisson(spec.events_per_day * duration_days)
    times = rng.integers(spec.start_s, spec.end_s, size=n_events) if n_events > 0 else []

    events = [
        Event(
            t=int(t),
            src_user=spec.actor_user,
            dst_user=spec.actor_user,
            src_comp=spec.actor_host,
            dst_comp=spec.target_host,
            auth_type="Kerberos",
            logon_type="LogOn",
            orientation="LogOn",
            success=True,
            is_malicious=spec.label_as_malicious,
            source="synthetic_conditioning",
        )
        for t in sorted(times)
    ]
    return events


def inject_conditioning(events: list[Event], spec: ConditioningSpec) -> list[Event]:
    """Merge a conditioning campaign into an existing, time-sorted event
    list and return a new, still time-sorted list.

    `events` is not modified in place (a new, sorted list is returned),
    so a caller can cheaply produce both an "attack-only" and an
    "attack-plus-conditioning" variant of the same base stream for a
    paired RQ2 comparison, e.g.::

        base = synth.generate(cfg, episodes=[episode])
        spec = boiling_frog_scenario(cfg, episode)
        conditioned = attack.inject_conditioning(base, spec)
        # compare detector/policy performance on `base` vs `conditioned`
    """
    conditioning_events = generate_conditioning_events(spec)
    if not conditioning_events:
        return list(events)
    merged = list(events) + conditioning_events
    merged.sort(key=lambda e: e.t)
    return merged


def boiling_frog_scenario(
    episode: LMEpisode,
    conditioning_days: float = 10.0,
    events_per_day: float = 2.0,
    gap_seconds: int = 3600,
    label_as_malicious: bool = False,
    seed: int = 0,
) -> ConditioningSpec:
    """Build a `ConditioningSpec` that targets the same actor and the
    first hop of a given `LMEpisode`, ending `gap_seconds` before the
    episode starts.

    This is the convenience constructor every RQ2 experiment config
    actually uses: it guarantees the conditioning campaign's
    `actor_host`/`actor_user` exactly match the subsequent real attack
    (the scenario's premise -- one compromised foothold used for both),
    and that conditioning never overlaps the attack itself.

    Parameters
    ----------
    episode : LMEpisode
        The real lateral-movement episode this conditioning campaign
        precedes. `target_host` is set to `episode.target_hosts[0]`
        (the attacker's first real hop) -- the host the attacker most
        wants the detector to have already learned is unremarkable.
    conditioning_days : float
        Length of the conditioning campaign.
    gap_seconds : int
        Quiet gap enforced between the end of conditioning and the
        start of the real episode, so the two phases are never
        ambiguous about which window they fall in.
    """
    end_s = episode.start_s - gap_seconds
    start_s = end_s - int(conditioning_days * 86400)
    return ConditioningSpec(
        start_s=max(start_s, 0),
        end_s=max(end_s, 0),
        actor_user=episode.actor_user,
        actor_host=episode.actor_host,
        target_host=episode.target_hosts[0],
        events_per_day=events_per_day,
        label_as_malicious=label_as_malicious,
        seed=seed,
    )
