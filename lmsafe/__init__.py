"""
lmsafe -- Safe Continual Updating for Lateral-Movement Detection
==================================================================

Research code accompanying the paper:

    "Safe Continual Updating of Lateral-Movement Detectors
     under Benign Drift and Update Poisoning"

Problem framing
----------------
Enterprise lateral-movement (LM) detectors are usually trained once on a
benign period and then frozen, or periodically retrained on recent traffic
to keep up with legitimate change ("drift": new hosts, new users, new
working hours). This package treats the *updating mechanism itself* as an
attack surface: an adversary who can influence what the detector sees
during normal operation (a "boiling-frog" / conditioning attacker) can bias
future updates so that the detector slowly learns to treat the attacker's
own traffic pattern as normal, before the real lateral-movement step
happens.

Three research questions drive the design of every module here:

    RQ1  Drift cost      -- how much detection quality do different update
                             policies lose to *benign* drift alone, and how
                             fast do they recover?
    RQ2  Update poisoning -- how much detection quality do those same
                             policies lose when an adaptive attacker injects
                             conditioning edges before attacking?
    RQ3  Gated defense    -- does a score-gated, delayed-commit update
                             policy recover RQ1's benign-drift performance
                             *without* re-opening the RQ2 poisoning gap, at
                             the same fixed alert budget?

Package layout
---------------
    lmsafe.data      -- event streams -> labeled, time-windowed graphs
                         (lanl.py, optc.py, windows.py, schema.py)
    lmsafe.synth     -- synthetic auth-log generator with known ground
                         truth (drift + injectable LM episodes)
    lmsafe.detectors -- link-prediction anomaly scorers (count baseline,
                         small torch sequence/GNN model)
    lmsafe.drift     -- Page-Hinkley / ADWIN concept-drift detectors, used
                         only to *trigger* an update policy, never as the
                         anomaly score itself
    lmsafe.policies  -- the five update policies compared in the paper
                         (frozen, periodic, sliding+replay, drift-triggered,
                         gated)
    lmsafe.attack    -- conditioning / update-poisoning injector used
                         *only* inside this offline evaluation harness, to
                         measure policy robustness (RQ2). It never touches
                         a live network.
    lmsafe.metrics   -- event-level AUC/AP, Rec@B (recall at a fixed alert
                         budget), detection delay, and episode-level
                         bootstrap confidence intervals
    lmsafe.runner    -- orchestrates data -> policy loop -> detector ->
                         metrics for one experiment cell

See README.md for the full pipeline diagram and how each script under
scripts/ maps to RQ1/RQ2/RQ3.
"""

__version__ = "0.1.0"
