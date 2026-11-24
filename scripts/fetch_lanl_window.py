#!/usr/bin/env python3
"""
Small CLI for quick, interactive LANL streaming checks -- the "peek at
the link and the data before committing to a multi-hour run" tool,
meant for a Colab cell. `scripts/run_experiment.py` (with a `lanl.yaml`
config) is what actually runs a full experiment; this script exists for
the smoke-test step before that, and for materializing a cached
window-graph file "offline" (once) that `run_experiment.py`'s
`lanl.cache_path` can then reuse.

    export LMSAFE_LANL_BASE_URL='<signed link>'   # or let it prompt you

    # 1) smoke test (seconds): first lines of a file + the red-team file
    python scripts/fetch_lanl_window.py peek --file redteam --n 5
    python scripts/fetch_lanl_window.py peek --file auth --n 5

    # 2) save the small red-team labels as CSV
    python scripts/fetch_lanl_window.py redteam --out redteam.csv

    # 3) stream+window a time range and pickle the result for
    #    run_experiment.py's lanl.cache_path to pick up directly
    python scripts/fetch_lanl_window.py build-cache \\
        --end-s 2592000 --out lmsafe_cache/lanl_dev_windows.pkl
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from lmsafe.config import LanlConfig
from lmsafe.data import lanl, lanl_stream
from lmsafe.data.windows import iter_windows


def cmd_peek(a: argparse.Namespace) -> None:
    base_url = lanl_stream.resolve_base_url()
    url = lanl_stream.file_url(base_url, a.file)
    for i, ln in enumerate(lanl_stream.iter_lines(url)):
        print(ln.decode("utf-8", "replace"))
        if i + 1 >= a.n:
            break


def cmd_redteam(a: argparse.Namespace) -> None:
    base_url = lanl_stream.resolve_base_url()
    records = lanl.load_redteam_records(base_url)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "user", "src_comp", "dst_comp"])
        for r in records:
            w.writerow([r.t, r.user, r.src_comp, r.dst_comp])
    print(f"wrote {len(records)} red-team rows to {a.out} (the public release documents 749)")


#: Re-exported under this name for backward compatibility (and because
#: this module's own tests reference it directly); the real
#: implementation lives in `lmsafe.data.lanl` so `run_experiment.py`'s
#: `cache_path` checkpointing can share the exact same crash-safety
#: guarantee without duplicating it.
_atomic_pickle_dump = lanl.atomic_pickle_dump


def cmd_build_cache(a: argparse.Namespace) -> None:
    cfg = LanlConfig(
        end_s=a.end_s,
        start_s=a.start_s,
        train_cutoff_s=a.train_cutoff_s,
        only_remote_logon=not a.no_logon_filter,
    )
    base_url = lanl_stream.resolve_base_url()
    redteam = lanl.load_redteam_records(base_url)
    events_iter = lanl.stream_auth_events(cfg, base_url=base_url, redteam=redteam)
    train_events, test_events_iter = lanl.split_train_test(events_iter, cfg)

    train_w = list(iter_windows(train_events, cfg.window_seconds))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stream-build the (potentially many-hour) test-window list with
    # periodic checkpoints, instead of only writing `out` once at the
    # very end. A multi-hour real-LANL run (58 days = 1B+ raw lines) has
    # a real chance of being killed by a Colab disconnect, a Ctrl+C, or
    # a network hiccup before it reaches the final line -- and
    # `iter_windows`/`stream_auth_events` are plain generators with
    # nothing written to disk until this function's caller used to do
    # it in one shot at the end. Without checkpoints, a run interrupted
    # at 90% progress leaves *zero* usable output (this is exactly what
    # was reported: a 58-day run reached day 50.3 of 58, got Ctrl+C'd,
    # and Google Drive came up `total 0` because nothing had ever been
    # written). Checkpointing every `checkpoint_every` windows means an
    # interrupted run still leaves a near-complete, directly-usable
    # cache file at `out` -- `run_experiment.py` cannot tell a
    # checkpoint from a finished cache, since both are just a
    # `(train_w, test_w)` pickle.
    test_w: list = []
    last_checkpoint_t = time.time()
    for w in iter_windows(test_events_iter, cfg.window_seconds):
        test_w.append(w)
        if len(test_w) % a.checkpoint_every == 0:
            _atomic_pickle_dump((train_w, test_w), out)
            print(
                f"[build-cache] checkpoint: {len(train_w)} train windows + "
                f"{len(test_w)} test windows so far -> {out} "
                f"({time.time() - last_checkpoint_t:.0f}s since last checkpoint)",
                file=sys.stderr,
                flush=True,
            )
            last_checkpoint_t = time.time()

    _atomic_pickle_dump((train_w, test_w), out)
    print(
        f"wrote {len(train_w)} train windows + {len(test_w)} test windows to {out} "
        f"-- point scripts/run_experiment.py's config at lanl.cache_path: {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("peek", help="print the first N lines of a cyber1 file (link smoke test)")
    p.add_argument("--file", choices=lanl_stream.FILE_NAMES, default="auth")
    p.add_argument("--n", type=int, default=5)
    p.set_defaults(fn=cmd_peek)

    p = sub.add_parser("redteam", help="save the red-team ground truth as CSV")
    p.add_argument("--out", default="redteam.csv")
    p.set_defaults(fn=cmd_redteam)

    p = sub.add_parser("build-cache", help="stream+window a time range and pickle it for run_experiment.py")
    p.add_argument("--start-s", type=int, default=0)
    p.add_argument("--end-s", type=int, required=True)
    p.add_argument("--train-cutoff-s", type=int, default=147_600)
    p.add_argument("--no-logon-filter", action="store_true", help="disable the fair remote-LogOn filter (debugging only)")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=24,
        help=(
            "write a partial (train_w, test_w) checkpoint to --out every N test "
            "windows (default 24, roughly once per simulated day at the default "
            "1-hour window length), so a multi-hour real-LANL run that gets "
            "interrupted (Ctrl+C, a Colab disconnect, a network drop) still "
            "leaves a usable, directly-loadable cache file instead of nothing"
        ),
    )
    p.set_defaults(fn=cmd_build_cache)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
