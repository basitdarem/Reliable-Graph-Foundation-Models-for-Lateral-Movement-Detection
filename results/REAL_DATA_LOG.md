# Real-data run log (reconstructed from Colab console output)

This file tracks every real LANL/OpTC result the user obtained on Colab,
reconstructed from pasted cell output after repeated Colab runtime
resets made it unsafe to rely on the VM's disk alone. Treat this file,
not a lost `/content` filesystem, as the source of truth for what was
actually run and what it produced.

## Real LANL, 30-day dev cutoff (`configs/lanl_dev.yaml`)

**Status: COMPLETE, real result in hand.**

- Window cache build: read 509,483,355 raw LANL auth-log lines, kept
  190,666,758 after the fairness filter, 702 of the 749 known red-team
  rows fell inside the 30-day cutoff (public release documents 749
  total). Completed cleanly this run -- no reconnects, unlike the
  58-day attempt. Wrote 41 train windows + 680 test windows to
  `lmsafe_cache/lanl_dev_windows.pkl`.
- Full sweep (`CountBaselineDetector` x 5 update policies) completed
  and wrote `results/lanl_dev/results.csv` (reproduced at that path in
  this package from the pasted console table -- the original file
  itself was lost to a second Colab runtime reset immediately after,
  before it could be downloaded. The numbers below are exact, copied
  from the printed table; a handful of diagnostic-only columns from the
  full ~30-column CSV were cut off in the terminal and are not
  reconstructed here, as they are NaN for most policies anyway and are
  not used in any paper metric):

| policy | weighted_auc | weighted_ap | episode_hit_rate | mean_delay_windows | n_episodes |
|---|---|---|---|---|---|
| frozen | 0.894323 | 0.000023 | 0.10 | 1.500000 | 20 |
| periodic | 0.939024 | 0.000050 | 0.10 | 5.000000 | 20 |
| sliding_replay | 0.980723 | 0.000189 | 0.15 | 3.666667 | 20 |
| drift_triggered | 0.967772 | 0.000106 | 0.15 | 3.000000 | 20 |
| gated | 0.949400 | 0.000055 | 0.10 | 6.500000 | 20 |

Label this in the paper as: **real LANL auth logs, 30-day evaluation
window** (not "58-day" -- see "Known limitations" in the main README for
why the 58-day run was abandoned, and why 30 days already contains
100% of the known red-team ground truth).

Note on absolute AP scale: AP here (~1e-4) is far smaller than the
synthetic RQ1-3 numbers (~1e-2 to 1e-3) -- expected, not a bug: real
LANL has ~190M kept events across 680 test windows with only 20
episodes total, a much more extreme class imbalance than the synthetic
scenarios are tuned to. The weighted AUC (0.89-0.98) is the more
robust/comparable headline number here. `sliding_replay` has the best
AUC/AP on this real cut; `gated` sits mid-pack on raw AUC/AP but that is
expected -- RQ1 is not where `gated`'s case is made (see RQ3), and this
run has no poisoning/conditioning attack injected at all, so there is
nothing here for `gated`'s defense mechanism to earn credit for yet.

## Real LANL, 58-day full cutoff (`configs/lanl_full.yaml`)

**Status: ABANDONED. No results table was ever produced.**

Two attempts; both lost to Colab runtime instability before reaching
the point of computing any detection metric at all (window-building
only, never got to run the 5 policies):
- Attempt 1: 6 network reconnects over ~7.4 hours, manually stopped
  (Ctrl+C) at day 40.64 of 58, checkpoint cache
  `lmsafe_cache/lanl_full_windows.pkl` (41 train + 912 test windows)
  reached day 40.64 -- then the entire Colab VM was recycled
  (`/content/lmsafe_project` no longer existed on reconnect) before the
  cache could be downloaded, losing it entirely.
- Decision: do not retry the 58-day cut. Use the 30-day dev cut above
  as the official real-LANL result instead -- it already covers 100% of
  the known red-team ground truth (all 749 labeled events occur before
  day 30), so the only thing the missing 28 days would add is exact
  numeric comparability with LMDEval's published 58-day protocol, not
  additional attacks.

## Synthetic RQ1 / RQ2 / RQ3 (controlled simulation, not real data)

Confirmed via the user's own Colab console output to match this
package's local `results/rq1/rq1_results.csv`,
`results/rq2/rq2_results.csv`, `results/rq3/rq3_results.csv` exactly
(same seed, deterministic). See the main README's "Known limitations"
and the chat discussion for why these three are controlled-simulation
results, not real-data results, and why that is methodologically
expected and defensible for RQ1-3 specifically.

## Cross-network transfer (synthetic network A -> synthetic network B)

`scratch` AUC=0.676085 AP=0.005047 rec@10=0.333 hit_rate=0.667;
`zero_shot` AUC=0.521148 AP=0.004749 rec@10=0.233 hit_rate=0.500;
`finetune` AUC=0.800455 AP=0.008792 rec@10=0.633 hit_rate=0.833.
Confirmed twice independently (once in this project's own fresh-kernel
notebook verification, once in the user's own Colab run) -- identical
numbers both times (deterministic, fixed seed).

## Real OpTC (`ecar-bro/short` sample)

**Status: mechanism verified, no usable score yet.** Default filter:
zero internal-to-internal windows (this specific sample is all outbound
traffic to external IPs). `--network-b-optc-keep-external`: full
pipeline runs end to end (1 train + 1 test window) but every
malicious-label metric is NaN (no red-team label join implemented yet).
Not a reportable number -- proof that the mechanism works on real data,
nothing more. Looking for a sample from `ecar-bro/evaluation` next (in
progress as of this log).

## Real OpTC breakthrough: `ecar` (host-based) vs `ecar-bro` (network-sensor)

**Key finding**: `ecar-bro` (Bro/Zeek network-sensor FLOW logs) structurally
contains almost no internal-to-internal traffic -- tested across both the
`short` sample and 15 combined files from the actual red-team day
(`23Sep19-red`, 1,046,144 FLOW records, 314 resolved hosts): **zero**
internal events in every case. Working hypothesis: `ecar-bro`'s sensor
is positioned at the network perimeter/egress, so it only sees
internal-to-external traffic, never east-west (internal-to-internal)
traffic -- regardless of which day or how many host-range files are
combined.

**Fix**: use `ecar` (the host-based sensor, not `ecar-bro`) instead.
Same `object=="FLOW"` event type exists inside `ecar`'s richer event
stream (which also has PROCESS/FILE/REGISTRY/MODULE/SHELL/TASK/
USER_SESSION records mixed in -- `lmsafe.data.optc`'s existing reader
already filters to `object=="FLOW"` only, so it works on `ecar` files
with **no code changes**). Combining 15 partial downloads (10 MB each)
from `ecar/evaluation/23Sep19-red/` (the same red-team day, now from
the host-based sensor):

- Total FLOW records: 1,924,871
- Total resolved hosts: 786
- **Total internal-to-internal events: 278,906**

This is the first real OpTC sample with substantial, genuine
internal/lateral-movement-relevant traffic. Files are large (~2 GB each
full size; only partial 10 MB downloads were used for diagnostics) --
being uploaded to the user's own Google Drive (via Drive API, not
`drive.mount`, which has proven unreliable in this project) for
permanent storage, folder name `lmsafe_real_optc_23Sep19_red_sample`.

Next: run `scripts/run_cross_network_transfer.py` against these 15
files to confirm the pipeline builds many real train/test windows (not
just 1-2 like the `ecar-bro` sample). Metrics will still read NaN until
the red-team ground-truth label join is implemented (see
`OpTCRedTeamGroundTruth.pdf`, not yet parsed) -- but window-building
and graph construction on substantial real internal traffic is itself a
meaningful new milestone.

## Red-team ground-truth join: implemented

`OpTCRedTeamGroundTruth.pdf` was fetched directly from the
`FiveDirections/OpTC-data` GitHub repo and read in full (7 pages). It
documents three engagement days against the `23Sep19`/`24Sep19`/`25Sep19`
dataset this project is already using, as a narrative log plus a
compromised-host list per day -- not event-level labels:

- **Day 1 ("Plain PowerShell Empire"), 2019-09-23 11:23:29-15:30:00
  (assumed US Eastern, matching the eCAR records' own "-04:00" offset):**
  Sysclient0201 -> Sysclient0402 -> Sysclient0660 -> DC1 (manual
  PowerShell Empire pivot chain), then DC1 spreads via WMI to 14 more
  hosts (SYSCLIENT0104/0170/0205/0255/0321/0355/0419/0462/0503/0559/
  0609/0771/0874/0955). **18 hosts total -- this is the exact day the
  `ecar/evaluation/23Sep19-red/` download already in progress covers.**
- **Day 2 ("Custom PowerShell Empire"), 2019-09-24 10:28:56 to
  2019-09-25 10:00:00:** phishing -> Sysclient0501/Sysclient0811 ->
  DC1 (DeathStar-automated) -> RDP pivot chain through Sysclient0974 ->
  Sysclient0005 (3.5 GB exfil) -> overnight spread to Sysclient0010/
  0069/0203/0358/0618/0851. 11 hosts total.
- **Day 3 ("Malicious Upgrade"), 2019-09-25 10:29:42-14:24:03:**
  hijacked Notepad++ auto-update (Meterpreter) on Sysclient0051 and
  Sysclient0351. 2 hosts total.

Implemented as `lmsafe/data/optc_groundtruth.py` (hostname
canonicalization to collapse the PDF's inconsistent spellings --
"Sysclient0201"/"SYSCLIENT0104"/"SysClient0201.systemia.com" all map to
one token -- plus the three day/host/time-window tuples above) and
wired into `lmsafe/data/optc.py` via a new opt-in
`OptcConfig.label_with_ground_truth` flag (default `False`, so every
existing test and caller is unaffected) and
`scripts/run_cross_network_transfer.py --network-b-optc-label-ground-truth`.
12 new tests added (144 total, all passing) covering canonicalization,
in-window/out-of-window matching, cross-day non-matching, the DC1
spelling difference between day 1 and day 2, and the opt-in default.

**Honest caveat, stated here and in the code/README:** this is a
**host+day** join, not LANL's **event-level** one -- the public OpTC
release does not ship per-event labels, only this narrative report.
Any traffic between two Day-1-compromised hosts during the Day-1 window
is labeled malicious, including ordinary unrelated benign traffic
between them. Report resulting metrics as an upper bound on precision,
not as directly comparable to LANL's event-level numbers.

**Next step:** once the in-progress 3-full-file `23Sep19-red` download
finishes and confirms real multi-window temporal coverage, re-run
`scripts/run_cross_network_transfer.py` with both
`--network-b-optc-keep-external` (or without, if enough internal
traffic resolves) and the new `--network-b-optc-label-ground-truth`
flag -- this is the run expected to finally produce a non-NaN real-OpTC
AUC/AP/episode-hit-rate table, since Day 1's 18 compromised hosts are
exactly the population `23Sep19-red` covers.
