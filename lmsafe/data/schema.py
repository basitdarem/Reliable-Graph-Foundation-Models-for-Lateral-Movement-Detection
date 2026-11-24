"""
Canonical event representation shared by every data source.

Every reader in this package (LANL, OpTC, the synthetic generator) emits a
stream of `Event` records in this one shape. Nothing downstream of
`lmsafe.data` -- not `windows.py`, not a detector, not a policy, not
`metrics.py` -- is allowed to know which dataset produced an Event. That
boundary is what makes it possible to swap LANL for OpTC or for the
synthetic generator without touching a single line of model or evaluation
code, and it is also what makes the synthetic generator a meaningful stand-
in for unit-testing the rest of the pipeline offline.

Design note on `is_malicious`
------------------------------
`is_malicious` is carried on the Event itself purely as *ground truth for
evaluation*. No code path in `lmsafe.detectors` or `lmsafe.policies` is
permitted to read it -- it is consumed exclusively by `lmsafe.metrics` and
by `lmsafe.attack` (which, by construction, needs to know which synthetic
events it itself injected). A unit test in tests/test_no_label_leak.py
grep-checks this invariant across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Event:
    """One authentication / lateral-movement-relevant event.

    Attributes
    ----------
    t : int
        Event time in seconds since the dataset's own epoch (LANL-style).
    src_user : str
        The authenticating principal, e.g. "U66@DOM1".
    dst_user : str
        The identity the authentication resolves to on the destination
        (often identical to src_user; kept distinct because LANL carries
        a separate column -- see `lmdeval`'s extraction logic in
        NOTICE.md -- and because impersonation shows up as a mismatch).
    src_comp : str
        Source computer/host.
    dst_comp : str
        Destination computer/host. The directed edge (src_comp, dst_comp)
        at time t is the unit of analysis for lateral-movement detection.
    auth_type : str
        Raw authentication protocol/type field from the source dataset
        (e.g. "Kerberos", "NTLM"); kept for diagnostics, not filtered on
        by default (see lmsafe.config.LanlConfig.filter_ntlm_only).
    logon_type : str
        Raw logon-type field: the Windows authentication *mechanism*
        (e.g. "Network", "Interactive", "Batch", "Service",
        "NetworkCleartext", "Unlock"). This is never "LogOn"/"LogOff" --
        that distinction lives in `orientation`, not here. Kept for
        diagnostics; not used by `is_remote_logon`.
    orientation : str
        LANL's own LogOn/LogOff marker for the event (values "LogOn" or
        "LogOff"). The fair protocol keeps only "LogOn"-oriented events
        between two different hosts -- see `is_remote_logon`.
    success : bool
        Whether the authentication succeeded.
    is_malicious : bool
        Ground-truth label. False for every real LANL/OpTC event unless it
        matches a red-team record; set by the synthetic generator and by
        `lmsafe.attack` for injected events. **Evaluation-only** -- see
        the module docstring.
    source : str
        Which reader produced this event: "lanl", "optc", or "synthetic".
        Lets a window-graph builder be dataset-agnostic while a debugging
        session can still ask "where did this edge come from?".
    """

    t: int
    src_user: str
    dst_user: str
    src_comp: str
    dst_comp: str
    auth_type: str = ""
    logon_type: str = ""
    orientation: str = ""
    success: bool = True
    is_malicious: bool = False
    source: str = "unknown"

    @property
    def edge(self) -> tuple[str, str]:
        """The directed host-level edge this event contributes to."""
        return (self.src_comp, self.dst_comp)

    def is_remote_logon(self) -> bool:
        """True for a remote LogOn between two *different* hosts.

        This is the fair-protocol filter (see lmsafe.config and
        NOTICE.md): it keeps "LogOn"-*oriented* events where the source
        host differs from the destination host, with no further
        filtering by auth_type, account type, or success. Deliberately
        *not* restricted to NTLM (CyberGFM-style preprocessing filters to
        NTLM-only and reports headline numbers on that easier,
        non-comparable subset).

        Checks `orientation`, not `logon_type`: in LANL's own auth.txt.gz
        schema, "LogOn"/"LogOff" is the value of the *orientation*
        column. `logon_type` instead holds the Windows logon mechanism
        ("Network", "Batch", "Interactive", ...) and is never literally
        "LogOn" -- comparing `logon_type` against "LogOn" silently
        rejects every real-data row (this was a real, shipped bug: it
        produced `kept=0` across 1.5B+ real LANL lines before being
        caught and fixed).
        """
        return self.orientation == "LogOn" and self.src_comp != self.dst_comp


#: CSV column order used by every CSV reader/writer in this package, kept
#: as one constant so a schema change is a one-line edit.
CSV_FIELDS: tuple[str, ...] = (
    "t",
    "src_user",
    "dst_user",
    "src_comp",
    "dst_comp",
    "auth_type",
    "logon_type",
    "orientation",
    "success",
    "is_malicious",
    "source",
)


def event_from_csv_row(row: dict) -> Event:
    """Parse one `csv.DictReader` row (strings) into an `Event`."""
    return Event(
        t=int(row["t"]),
        src_user=row["src_user"],
        dst_user=row["dst_user"],
        src_comp=row["src_comp"],
        dst_comp=row["dst_comp"],
        auth_type=row.get("auth_type", ""),
        logon_type=row.get("logon_type", ""),
        orientation=row.get("orientation", ""),
        success=_parse_bool(row.get("success", "True")),
        is_malicious=_parse_bool(row.get("is_malicious", "False")),
        source=row.get("source", "unknown"),
    )


def event_to_csv_row(e: Event) -> dict:
    """Serialize an `Event` to a `dict` suitable for `csv.DictWriter`."""
    return {
        "t": e.t,
        "src_user": e.src_user,
        "dst_user": e.dst_user,
        "src_comp": e.src_comp,
        "dst_comp": e.dst_comp,
        "auth_type": e.auth_type,
        "logon_type": e.logon_type,
        "orientation": e.orientation,
        "success": e.success,
        "is_malicious": e.is_malicious,
        "source": e.source,
    }


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "t", "yes")


@dataclass(frozen=True)
class RedTeamRecord:
    """One ground-truth red-team (lateral-movement) record from
    redteam.txt.gz: ``time,user,src_comp,dst_comp``."""

    t: int
    user: str
    src_comp: str
    dst_comp: str

    def key(self) -> tuple[int, str, str, str]:
        """Join key against an Event: (t, dst_user, src_comp, dst_comp).

        This mirrors LMDEval's `extract_lanl.py` exactly: the red-team
        file's "user" field is matched against an auth event's
        *destination* user, not its source user. Getting src/dst
        reversed here silently produces a detector that never sees a
        correctly labeled positive -- see NOTICE.md and
        tests/test_lanl_labels.py for a regression test.
        """
        return (self.t, self.user, self.src_comp, self.dst_comp)
