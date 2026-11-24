"""
OpTC adapter -- real implementation, built from a confirmed sample.

Status: this module was a documented stub until a real sample of
`ecar-bro/short` (241 FLOW records, one host-range partition, 2019-09-25)
was obtained and inspected line-by-line to pin down the exact schema
(see this module's history / project NOTICE for the earlier stub). The
field names and behavior below are confirmed against that real sample,
not guessed from the generic public eCAR documentation -- notably:

- The real `action` value for a FLOW record is ``"INFO"``, not
  ``"START"``/``"CREATE"`` as the generic eCAR docs suggest for other
  object types. This reader filters on ``object == "FLOW"`` alone and
  does not additionally filter on `action`.
- `timestamp` is ISO-8601 with a UTC offset and a **variable-width**
  fractional-seconds part -- 0, 1, 2, 3, or more digits all appear in
  real data (e.g. ``"...T09:56:52-04:00"`` with none at all and
  ``"...T09:07:45.26-04:00"`` with exactly two). `datetime.fromisoformat`
  is not reliably lenient about this across Python versions, so
  `_parse_optc_timestamp` below hand-parses it with a regex instead.
- A FLOW record does not carry a destination *hostname* -- only a
  `dest_ip` (and, for the local, reporting host, a `src_ip`). There is
  no analogue of LANL's `src_comp,dst_comp` pair sitting directly in one
  record. See "IP-to-host resolution" below for how this reader
  reconstructs host-to-host edges anyway.

Data source and known labeling caveat
--------------------------------------
OpTC is distributed by the FiveDirections project as a set of `ecar`
JSON-lines files (flow/process/file events) on a shared Google Drive
folder (https://drive.google.com/drive/u/0/folders/1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa),
roughly 1 TB uncompressed across all hosts and days -- too large to load
in full in this project's sandbox. This reader targets the subset that
is actually obtainable and useful here: one or a few `ecar-bro/*` files
(each one host-range partition's FLOW-START events, optionally
gzip-compressed, auto-detected). It is **not** a streaming reader like
`lmsafe.data.lanl_stream` -- every given file is fully parsed into
memory and the results sorted by time, because `ecar-bro` is partitioned
by host range rather than globally time-ordered, so a true single-pass
stream across many files would need an external k-way merge this
project does not yet implement. This is fine at the "a handful of real
sample files" scale this project operates at; it would need revisiting
before pointing this reader at the full ~1 TB release.

The project's own literature review flags an important fairness issue
here too: most published OpTC results use *host-based* red-team labels
(a host is "malicious" for the whole engagement day), which is a much
easier target than LANL's *event-level* labels. The fair protocol
(Larroche/LMDEval-style) instead uses an event-level hybrid manual
labeling released alongside LMDEval. This reader now optionally joins
against `lmsafe.data.optc_groundtruth`, a direct transcription of
`OpTCRedTeamGroundTruth.pdf` (see that module's docstring for the full
per-day host list and the honest host+day-not-event-level caveat) --
pass `OptcConfig(label_with_ground_truth=True)` to turn it on. Left at
its default (`False`), every `Event` still has `is_malicious=False`
exactly as before, so existing callers and the cross-network-transfer
chapter's unsupervised use are unaffected.

IP-to-host resolution
----------------------
A FLOW record's `properties.direction` is `"outbound"` or `"inbound"`
(confirmed in the sample). When `"outbound"`, the *reporting* host
(`hostname`) is the connection's source, so its own IP is `src_ip` and
the far end is `dest_ip`; when `"inbound"` the reverse holds. Since
every host in the engagement reports its own outbound/inbound FLOW
records, a complete self-consistent `ip -> hostname` table can be built
by watching, for every record, which IP belongs to the reporting host
itself -- no separate asset inventory is needed. `build_ip_host_table`
does this (majority vote per IP, to tolerate the rare DHCP-style
reassignment or logging glitch). A FLOW's *other* endpoint is then
resolved through that same table: if it resolves to a known host, the
record contributes a genuine host-to-host edge; if it does not (an
external IP, e.g. a C2 server or ordinary internet traffic), the record
is dropped by default (`OptcConfig.only_internal_internal=True`) -- the
OpTC-equivalent of LANL's "remote LogOn between two *different* hosts"
fairness filter in `lmsafe.data.lanl.Event.is_remote_logon`. Every
emitted `Event` sets `orientation="LogOn"` unconditionally, so the
*same* `is_remote_logon()` check LANL events are graded through also
accepts these -- no change needed anywhere downstream of `schema.Event`
to run OpTC through the existing windows/detector/policy/metrics
pipeline, exactly as this module's original stub promised.

Field mapping into `schema.Event`
-----------------------------------
t           : parsed timestamp, in integer seconds relative to the
              *earliest* timestamp across all records passed to
              `stream_optc_events` in this call (LANL-style "seconds
              since the dataset's own epoch").
src_user    : `properties.principal` of the record where it was
              observed -- there is no separate source/destination
              identity in a network flow, so...
dst_user    : ...this is set identical to `src_user` (see schema.py's
              own docstring: "often identical to src_user" is an
              expected, documented case, not a placeholder bug).
src_comp    : resolved source host (see IP resolution above).
dst_comp    : resolved destination host.
auth_type   : repurposed to carry `properties.dest_port` (string) --
              OpTC's nearest analogue of "what mechanism" is which port
              a connection targets (445=SMB, 3389=RDP, 22=SSH, ... are
              the classic lateral-movement-relevant ones); kept for
              diagnostics/ablations exactly as LANL's own `auth_type`
              is, never filtered on by default.
logon_type  : repurposed to carry `properties.direction`
              ("outbound"/"inbound" as seen on the *reporting* host's
              own record, before resolution) -- diagnostics only.
orientation : always `"LogOn"` for every emitted event (see above).
success     : always `True` -- a Bro/Zeek FLOW-START record implies a
              connection was actually established; there is no
              separate failure signal for this object type in the
              sample inspected.
is_malicious: always `False` (see labeling caveat above).
source      : `"optc"`.
"""
from __future__ import annotations

import collections
import dataclasses
import gzip
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional

from . import optc_groundtruth
from .schema import Event

_TS_RE = re.compile(
    r"^(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})T"
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})"
    r"(?:\.(?P<frac>\d+))?"
    r"(?P<tzsign>[+-])(?P<tzh>\d{2}):(?P<tzm>\d{2})$"
)


def _parse_optc_timestamp(ts: str) -> float:
    """Parse one OpTC `timestamp` string into a UTC epoch float.

    Hand-rolled instead of `datetime.fromisoformat` because real OpTC
    timestamps have a *variable-width* fractional-seconds part (0 to at
    least 3 digits observed in a real sample -- e.g. both
    ``"09:56:52-04:00"`` with none and ``"09:07:45.26-04:00"`` with
    exactly two), which `fromisoformat` does not reliably accept across
    Python versions (it is strict about 3-or-6-digit microseconds on
    many versions). Raises `ValueError` on anything that does not match
    the confirmed real shape.
    """
    m = _TS_RE.match(ts)
    if not m:
        raise ValueError(f"unrecognized OpTC timestamp format: {ts!r}")
    frac = m.group("frac") or "0"
    micros = int(frac.ljust(6, "0")[:6])
    sign = 1 if m.group("tzsign") == "+" else -1
    offset = timedelta(hours=int(m.group("tzh")), minutes=int(m.group("tzm"))) * sign
    dt = datetime(
        int(m.group("y")), int(m.group("mo")), int(m.group("d")),
        int(m.group("h")), int(m.group("mi")), int(m.group("s")),
        micros, tzinfo=timezone(offset),
    )
    return dt.timestamp()


@dataclasses.dataclass(frozen=True)
class OptcConfig:
    """Configuration for one OpTC extraction run. Duck-type compatible
    with `lmsafe.data.lanl.split_train_test` (which only ever reads
    `.train_cutoff_s` off whatever config it is given), so that function
    is reused here unchanged rather than re-implemented."""

    window_seconds: int = 3600
    #: Train/test boundary, in seconds from the first event seen in this
    #: call. OpTC's engagement window is much shorter than LANL's (days,
    #: not months); there is no single project-wide default here the way
    #: `LANL_TRAIN_CUTOFF_S` is -- callers should set this deliberately
    #: from the actual span of the files they pass in.
    train_cutoff_s: int = 2 * 3600
    #: The OpTC-equivalent of LanlConfig.only_remote_logon: keep only
    #: FLOW records whose *other* endpoint also resolves to a known
    #: internal host. Keeping this True is what makes the fairness
    #: argument in this module's docstring hold.
    only_internal_internal: bool = True
    #: When True, join every event against `optc_groundtruth.is_malicious`
    #: (host+day red-team ground truth transcribed from
    #: `OpTCRedTeamGroundTruth.pdf`) instead of leaving `is_malicious`
    #: hardcoded False. Default False preserves every existing caller's
    #: behavior exactly. See `optc_groundtruth`'s docstring for the
    #: host+day-not-event-level caveat before reporting a metric computed
    #: with this on.
    label_with_ground_truth: bool = False


class _FlowRecord(NamedTuple):
    t: float
    hostname: str
    direction: str
    src_ip: str
    dest_ip: str
    dest_port: str
    principal: str


def _open_maybe_gzip(path: Path):
    """Open `path` for text reading, transparently handling gzip
    (detected by magic bytes, not by file extension -- a real sample
    file from the shared Drive folder may or may not carry a `.gz`
    suffix depending on how it was downloaded)."""
    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _iter_flow_records(paths: Iterable[Path]) -> Iterator[_FlowRecord]:
    """Parse every FLOW-object JSON line across all `paths` into a
    `_FlowRecord`. Lines that fail to parse as JSON, are not FLOW
    objects, or are missing a required field are silently skipped (an
    eCAR release is known to contain occasional malformed lines -- see
    errata.md on the FiveDirections repo); this mirrors
    `lmsafe.data.lanl.stream_auth_events`'s own tolerance of bad lines.
    """
    for path in paths:
        with _open_maybe_gzip(Path(path)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("object") != "FLOW":
                    continue
                props = rec.get("properties") or {}
                hostname = rec.get("hostname")
                direction = props.get("direction")
                src_ip = props.get("src_ip")
                dest_ip = props.get("dest_ip")
                ts = rec.get("timestamp")
                if not (hostname and direction and src_ip and dest_ip and ts):
                    continue
                try:
                    t = _parse_optc_timestamp(ts)
                except ValueError:
                    continue
                yield _FlowRecord(
                    t=t,
                    hostname=hostname,
                    direction=direction,
                    src_ip=src_ip,
                    dest_ip=dest_ip,
                    dest_port=str(props.get("dest_port", "")),
                    principal=rec.get("principal") or "",
                )


def build_ip_host_table(records: Iterable[_FlowRecord]) -> dict[str, str]:
    """Build an `ip -> hostname` table purely from self-observation: for
    an `"outbound"` record the reporting host's own IP is `src_ip`; for
    `"inbound"` it is `dest_ip`. Ties (the same IP observed against more
    than one hostname -- e.g. a DHCP reassignment during the capture
    window) are resolved by majority vote so one glitchy record cannot
    flip an otherwise-consistent mapping.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in records:
        if r.direction == "outbound":
            votes[r.src_ip][r.hostname] += 1
        elif r.direction == "inbound":
            votes[r.dest_ip][r.hostname] += 1
        # any other direction value: no self-IP signal to extract, skip
    return {ip: counter.most_common(1)[0][0] for ip, counter in votes.items()}


def stream_optc_events(paths: Iterable[Path], cfg: OptcConfig) -> list[Event]:
    """Read every FLOW record across `paths`, resolve host-to-host
    edges, and return a time-sorted `list[Event]`.

    Returns a materialized list, not a lazy iterator -- unlike
    `lmsafe.data.lanl.stream_auth_events`, this is not a single
    chronological file: `ecar-bro` partitions by host range, so records
    from different files must be merged and sorted by time before
    `lmsafe.data.windows.iter_windows` (which requires sorted input) can
    consume them. See this module's docstring for the scaling caveat.
    """
    paths = [Path(p) for p in paths]
    records = list(_iter_flow_records(paths))
    if not records:
        return []

    ip_table = build_ip_host_table(records)
    t0 = min(r.t for r in records)

    events: list[Event] = []
    for r in records:
        if r.direction == "outbound":
            src_comp = r.hostname
            dst_comp = ip_table.get(r.dest_ip)
        elif r.direction == "inbound":
            src_comp = ip_table.get(r.src_ip)
            dst_comp = r.hostname
        else:
            continue

        if dst_comp is None or src_comp is None:
            if cfg.only_internal_internal:
                continue
            # keep_external path: fall back to the raw IP as a
            # pseudo-hostname so the edge is still representable, rather
            # than silently losing the record.
            src_comp = src_comp or r.src_ip
            dst_comp = dst_comp or r.dest_ip

        if src_comp == dst_comp:
            # a host talking to itself is not a lateral-movement-
            # relevant edge; same spirit as LANL's own same-host check.
            continue

        malicious = False
        if cfg.label_with_ground_truth:
            # r.t (not the relativized t below) is the absolute UTC
            # epoch second this record was observed at -- the ground
            # truth's day windows are also absolute UTC, so the join
            # must happen before t is rebased to "seconds since t0".
            malicious = optc_groundtruth.is_malicious(src_comp, dst_comp, r.t)

        events.append(
            Event(
                t=int(r.t - t0),
                src_user=r.principal,
                dst_user=r.principal,
                src_comp=src_comp,
                dst_comp=dst_comp,
                auth_type=r.dest_port,
                logon_type=r.direction,
                orientation="LogOn",
                success=True,
                is_malicious=malicious,
                source="optc",
            )
        )

    events.sort(key=lambda e: e.t)
    return events
