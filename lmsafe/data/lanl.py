"""
Fair LANL extraction protocol.

This module is the one place that decides *which* rows of the LANL
auth.txt.gz stream become `Event`s, and *which* of those get
`is_malicious=True`. Both decisions follow the "fair evaluation protocol"
described in the project's literature review (adapted from Larroche's
LMDEval, BSD-2-Clause -- see NOTICE.md) specifically so results here are
comparable to it, and *not* comparable to papers (e.g. CyberGFM) that use
an easier random split, a 14-day window, an NTLM-only filter, or host-based
OpTC labels. That incomparability is a known, citable fact about this
literature, not an oversight -- see README.md "Why a fair split matters".

auth.txt.gz columns (LANL's own schema)::

    time,src_user@domain,dst_user@domain,src_comp,dst_comp,
    auth_type,logon_type,orientation,success

redteam.txt.gz columns::

    time,user,src_comp,dst_comp

Label join rule (from LMDEval's extract_lanl.py, reproduced in
`schema.RedTeamRecord.key`): a red-team record matches an auth event when
``(event.t, event.dst_user, event.src_comp, event.dst_comp)`` equals
``(record.t, record.user, record.src_comp, record.dst_comp)`` -- note the
red-team "user" matches the auth event's *destination* user, not its
source user.
"""
from __future__ import annotations

import dataclasses
import gzip
import os
import pickle
import sys
import time
from pathlib import Path
from typing import IO, Iterator, Optional

from ..config import LanlConfig
from . import lanl_stream
from .schema import Event, RedTeamRecord

AUTH_NCOLS = 9


def atomic_pickle_dump(obj, out: Path) -> None:
    """Write `obj` to `out` without ever leaving a half-written or
    corrupt file at that path: write to a sibling temp file first, then
    `os.replace` it into place (atomic on POSIX). Shared by
    `scripts/fetch_lanl_window.py` (`build-cache`) and
    `scripts/run_experiment.py` (`build_lanl_windows`'s `cache_path`
    checkpointing) -- both stream real LANL data for potentially hours,
    so both need every intermediate checkpoint write to be crash-safe: a
    checkpoint killed mid-write (Ctrl+C, a Colab disconnect, an OOM)
    must leave the *previous* good checkpoint intact at `out`, never a
    truncated pickle that fails to load."""
    out = Path(out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, out)


def _auth_fields_to_event(raw_fields: list[bytes]) -> Optional[Event]:
    """Parse one raw auth.txt.gz line (already split on b",") into an
    Event, or return None if the line is malformed.

    Malformed lines are counted by the caller (see `stream_auth_events`
    stats) rather than raising -- a handful of corrupt lines should not
    abort a multi-hour streaming read, but a silently *large* bad-line
    count is exactly the kind of thing that should be visible, which is
    why we return a count instead of swallowing it entirely.
    """
    if len(raw_fields) != AUTH_NCOLS or not raw_fields[0].isdigit():
        return None
    t_s, src_u, dst_u, src_c, dst_c, auth_t, logon_t, orient, success = raw_fields
    return Event(
        t=int(t_s),
        src_user=src_u.decode("utf-8", "replace"),
        dst_user=dst_u.decode("utf-8", "replace"),
        src_comp=src_c.decode("utf-8", "replace"),
        dst_comp=dst_c.decode("utf-8", "replace"),
        auth_type=auth_t.decode("utf-8", "replace"),
        logon_type=logon_t.decode("utf-8", "replace"),
        orientation=orient.decode("utf-8", "replace"),
        success=(success == b"Success"),
        source="lanl",
    )


def _raw_line_passes_fair_prefilter(raw: list[bytes], cfg: LanlConfig) -> bool:
    """Cheap, decode-free early-reject check on the still-raw `b","`-split
    auth.txt.gz fields, used by `stream_auth_events` to skip the costly
    utf-8 decode + `Event` construction + `apply_fair_protocol` call
    entirely for a line that would be discarded anyway.

    This is a *performance-only* shortcut, never a behavior change:
    `apply_fair_protocol` is still the sole authority on what gets kept,
    and still runs (on the full decoded `Event`) for every line this
    returns True for. The real-data fair-protocol filter rejects the
    majority of lines (same-host events, non-"LogOn"-oriented events --
    empirically ~63% of all real auth.txt.gz rows), and on a billion-row
    real run, decoding 7 string fields and constructing a dataclass for
    every one of those doomed-to-be-discarded rows is the single biggest
    avoidable cost in the whole pipeline.

    Column indices (see the module docstring / AUTH_NCOLS): 3=src_comp,
    4=dst_comp, 5=auth_type, 7=orientation.

    Why comparing *raw bytes* against literal ASCII strings is safe,
    i.e. provably never rejects a line `apply_fair_protocol` would have
    kept:

    * `b"LogOn"` / `b"NTLM"` are pure ASCII. A byte string decodes
      (utf-8, errors="replace") to exactly "LogOn"/"NTLM" if and only if
      it *is* exactly those ASCII bytes -- utf-8 is a unique encoding of
      ASCII text, and any decode error inserts U+FFFD, which can never
      collapse back to that literal ASCII string. So the orientation and
      auth_type comparisons below are *exact* equivalents of the
      decoded-string comparisons in `Event.is_remote_logon` /
      `apply_fair_protocol`, not approximations.
    * `raw[3] != raw[4]` (raw bytes) is a safe *over*-approximation of
      `src_comp != dst_comp` (decoded strings): if the raw bytes are
      identical, decoding is a deterministic function of the bytes, so
      the decoded strings are certainly identical too (never a false
      "different hosts"). The only direction that can theoretically
      diverge is two *different*, both-invalid-utf8 byte sequences each
      decoding to a replacement string that happens to coincide --
      vanishingly unlikely for the clean ASCII hostnames LANL actually
      publishes, and even if it occurred this function would simply
      fail to skip early (falling through to the real, authoritative
      check) rather than skip something it shouldn't have.
    """
    if cfg.only_remote_logon and not (raw[7] == b"LogOn" and raw[3] != raw[4]):
        return False
    if cfg.filter_ntlm_only and raw[5] != b"NTLM":
        return False
    return True


def load_redteam_records(
    base_url: Optional[str] = None,
    *,
    local_path: Optional[Path] = None,
) -> list[RedTeamRecord]:
    """Load every red-team ground-truth record.

    redteam.txt.gz is tiny (4.8 KB, 749 rows in the public release), so
    unlike auth.txt.gz we just read the whole thing into memory -- either
    from a local copy (`local_path`, e.g. the test fixture under
    tests/fixtures/redteam.txt.gz) or streamed from `base_url`.
    """
    records: list[RedTeamRecord] = []

    def _parse(lines: Iterator[bytes]) -> None:
        for ln in lines:
            parts = ln.strip().split(b",")
            if len(parts) != 4 or not parts[0].isdigit():
                continue
            records.append(
                RedTeamRecord(
                    t=int(parts[0]),
                    user=parts[1].decode("utf-8", "replace"),
                    src_comp=parts[2].decode("utf-8", "replace"),
                    dst_comp=parts[3].decode("utf-8", "replace"),
                )
            )

    if local_path is not None:
        with gzip.open(local_path, "rb") as fh:
            _parse(iter(fh))
    else:
        if base_url is None:
            base_url = lanl_stream.resolve_base_url()
        url = lanl_stream.file_url(base_url, "redteam")
        _parse(lanl_stream.iter_lines(url))

    return records


def stream_auth_events(
    cfg: LanlConfig,
    *,
    base_url: Optional[str] = None,
    redteam: Optional[list[RedTeamRecord]] = None,
    progress_every: int = 5_000_000,
    progress_stream: IO[str] = sys.stderr,
) -> Iterator[Event]:
    """Stream auth events in `[cfg.start_s, cfg.end_s]`, labeled against
    `redteam`, applying the fair-protocol filter.

    This is a generator: nothing is buffered beyond the current line and
    the (small) red-team lookup set, so memory use is O(len(redteam)),
    not O(file size). The caller controls how much of the 7.2 GB file is
    actually transferred by choosing `cfg.end_s` -- reading stops as soon
    as the stream passes `cfg.end_s` (plus a small slack to tolerate
    slightly-unsorted input), *not* after reading the whole file.

    Parameters
    ----------
    cfg : LanlConfig
        `cfg.only_remote_logon` applies the fair-protocol filter (keep
        LogOn events between two different hosts, don't restrict to
        NTLM). `cfg.filter_ntlm_only` exists only so a caller can
        deliberately reproduce the *non-fair* CyberGFM-style preprocessing
        for a side-by-side comparison; it defaults to False and the
        README explains why it should stay that way for the paper's
        headline numbers.
    base_url : str, optional
        Pre-resolved signed LANL link. If omitted, resolved via
        `lanl_stream.resolve_base_url()` (env var, else interactive
        prompt) -- never hardcoded.
    redteam : list[RedTeamRecord], optional
        Pre-loaded labels. If omitted, loaded via `load_redteam_records`
        using the same `base_url`.

    Yields
    ------
    Event
        With `is_malicious=True` set exactly for events whose
        `(t, dst_user, src_comp, dst_comp)` matches a red-team record.
    """
    if base_url is None:
        base_url = lanl_stream.resolve_base_url()
    if redteam is None:
        redteam = load_redteam_records(base_url)
    redteam_keys = {r.key() for r in redteam}

    url = lanl_stream.file_url(base_url, "auth")
    slack_s = SECONDS_SLACK_FOR_DISORDER

    # Local variables, not a dict, for the four counters touched on
    # *every* line of a billion-plus-line real run: a dict-key lookup
    # (hash + compare) costs measurably more per access than a local
    # variable slot in CPython, and this loop does four to six such
    # accesses per line. `stats` is only reassembled (cheaply, once)
    # right before each progress print and at the very end.
    read = kept = bad = malicious = last_t = 0
    t0 = time.time()
    end_s = cfg.end_s
    start_s = cfg.start_s
    for ln in lanl_stream.iter_lines(url):
        read += 1

        # Report progress by *raw lines read*, independent of how many of
        # them end up filtered out below. The naive version of this check
        # used to live after the filtering (`only_remote_logon`, time
        # window, etc.) and fired on `kept % progress_every`-style logic
        # gated behind several `continue`s -- since most raw auth lines
        # are filtered out (same-host LogOffs, non-LogOn auth types,
        # ...), the one specific line where the raw count crossed a
        # progress_every boundary would often be exactly one of those
        # filtered lines, silently skipping that round's print instead of
        # merely delaying it. Over a multi-hour real stream that produced
        # long, unpredictable stretches with no output at all even though
        # the stream was reading fine -- indistinguishable from a hang.
        # Checking right here, unconditionally, guarantees a print every
        # `progress_every` raw lines no matter what fraction gets kept.
        if read % progress_every == 0:
            print(
                f"[lanl] read {read:,} lines, last t={last_t:,}s "
                f"({last_t / 86400:.2f} d), kept {kept:,}, "
                f"bad {bad:,}, malicious {malicious}, "
                f"{time.time() - t0:.0f}s elapsed",
                file=progress_stream,
                flush=True,
            )

        raw = ln.split(b",")
        if len(raw) != AUTH_NCOLS or not raw[0].isdigit():
            bad += 1
            continue
        t = int(raw[0])
        last_t = t
        if t > end_s + slack_s:
            break  # past the window -- stop reading, do not scan the rest
        if t > end_s or t < start_s:
            continue

        # Fast, decode-free reject: skip straight to the next line
        # without ever building an Event for one the fair protocol
        # would discard anyway (see `_raw_line_passes_fair_prefilter`'s
        # docstring for the correctness argument). This is the dominant
        # cost saving on real data, where the large majority of rows
        # fail this check.
        if not _raw_line_passes_fair_prefilter(raw, cfg):
            continue

        parsed = _auth_fields_to_event(raw)
        if parsed is None:
            bad += 1
            continue

        ev = apply_fair_protocol(parsed, redteam_keys, cfg)
        if ev is None:
            continue
        if ev.is_malicious:
            malicious += 1
        kept += 1

        yield ev

    print(
        f"[lanl] done: read {read:,} | kept {kept:,} | "
        f"bad {bad} | malicious {malicious} "
        f"(redteam has {len(redteam)} total records)",
        file=progress_stream,
        flush=True,
    )


#: How far past `end_s` to keep reading before giving up, to tolerate a
#: handful of out-of-order lines near the boundary (matches the tested
#: standalone script's default).
SECONDS_SLACK_FOR_DISORDER = 3600


def apply_fair_protocol(
    ev: Event, redteam_keys: set[tuple[int, str, str, str]], cfg: LanlConfig
) -> Optional[Event]:
    """Pure function: apply the fair-protocol filters to one parsed
    `Event` and attach its ground-truth label.

    Pulled out of `stream_auth_events` specifically so it can be unit
    tested (tests/test_lanl_labels.py) against hand-built Events and a
    small red-team set, with no network access and no dependence on
    `lanl_stream` at all. This is the one function that encodes "what
    counts as a kept event" and "what counts as a labeled positive" --
    exactly the two decisions a wrong-by-one-field-order bug could get
    wrong silently (see the docstring on `RedTeamRecord.key`).

    Returns
    -------
    Event or None
        A copy of `ev` with `is_malicious` set correctly, or None if
        `ev` is filtered out by the fair protocol (not a remote LogOn,
        or -- only if the caller opted into the non-fair comparison
        mode -- not NTLM).
    """
    if cfg.only_remote_logon and not ev.is_remote_logon():
        return None
    if cfg.filter_ntlm_only and ev.auth_type != "NTLM":
        return None
    is_mal = (ev.t, ev.dst_user, ev.src_comp, ev.dst_comp) in redteam_keys
    if is_mal == ev.is_malicious:
        return ev
    # dataclass is frozen -- return a relabeled copy rather than mutate.
    return dataclasses.replace(ev, is_malicious=is_mal)


def split_train_test(
    events: Iterator[Event], cfg: LanlConfig
) -> tuple[list[Event], Iterator[Event]]:
    """Materialize the benign training events (t < cfg.train_cutoff_s)
    into a list, and return the remaining events as a (still-lazy)
    iterator for the test stream.

    Why training is materialized but test is not: the training window is
    small (41 hours) and every policy needs repeated access to it (to
    fit the initial model, and as the "clean anchor memory" for the gated
    policy in lmsafe.policies). The test stream can span up to 58 days and
    is only ever consumed once, window by window, by `lmsafe.runner` --
    so it stays an iterator and is never fully materialized.

    Caution: this function consumes the input iterator lazily and
    correctly only if `events` is sorted by time ascending, which both
    `stream_auth_events` and `lmsafe.synth.generate` guarantee.
    """
    train: list[Event] = []
    first_test_event: Optional[Event] = None

    # Drain the strictly-training prefix eagerly so `train` is a
    # complete, ready-to-use list for the caller immediately. Stop at
    # the first event at or after the cutoff and hold onto it, since
    # calling next() on `events` to peek already consumed it from the
    # underlying stream -- it must be re-yielded, not dropped.
    for ev in events:
        if ev.t < cfg.train_cutoff_s:
            train.append(ev)
        else:
            first_test_event = ev
            break

    def _test_stream() -> Iterator[Event]:
        if first_test_event is not None:
            yield first_test_event
        yield from events

    return train, _test_stream()
