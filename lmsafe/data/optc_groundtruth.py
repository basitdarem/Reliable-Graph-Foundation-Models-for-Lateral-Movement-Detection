"""
Red-team ground truth for the OpTC dataset.

Source: `OpTCRedTeamGroundTruth.pdf`, published by FiveDirections in the
`FiveDirections/OpTC-data` GitHub repository alongside the `ecar` /
`ecar-bro` / `bro` data folders (DISTRIBUTION A, approved for public
release, per the document's own footer). This module is a direct
transcription of that document's two facts per engagement day: which
hosts the red team used, and the day's first-to-last recorded action
timestamp. It is not derived from any inference about the data itself.

Three engagement days are documented:

- Day 1, "Plain PowerShell Empire" (2019-09-23): manual PowerShell
  Empire use, pivoting Sysclient0201 -> Sysclient0402 -> Sysclient0660
  -> the domain controller, then spreading to 14 more hosts via WMI.
- Day 2, "Custom PowerShell Empire" (2019-09-24 to 2019-09-25 10:00):
  phishing -> DeathStar-automated domain compromise -> RDP pivoting
  chain -> data exfiltration, plus an overnight WMI-subscription
  persistence re-check-in the next morning.
- Day 3, "Malicious Upgrade" (2019-09-25): a hijacked Notepad++
  auto-update delivering a Meterpreter payload on two hosts.

Known limitation (stated plainly, not hidden): this is a *host+day*
ground truth, not LANL's *event-level* one, because the public OpTC
release does not ship event-level labels -- only this narrative report
of which hosts were used and when. The approach taken here (mark a FLOW
event malicious when BOTH of its hosts are on that day's compromised
list AND its timestamp falls inside that day's action window) is the
same host-day labeling convention flagged in the lateral-movement-
detection literature as the common, but overly permissive, way to label
OpTC: see "On fair and realistic performance evaluations for graph-based
lateral movement detectors" (Larroche et al.) for why this over-labels
relative to a true event-level ground truth -- ordinary, unrelated
benign traffic between two compromised hosts during the engagement
window is also marked malicious here, inflating apparent recall/AUC and
likely deflating apparent precision relative to the true rate. Any
OpTC-based malicious-label metric produced with this module should be
reported with that caveat, not presented as directly comparable to a
LANL metric on equal footing.

Time zone note: the PDF's log timestamps carry no explicit UTC offset
("09/23/19 11:23:29"). The eCAR FLOW records themselves do carry one,
and every real sample this project has inspected uses "-04:00" (US
Eastern Daylight Time, the correct offset for Sep 2019). This module
assumes the ground truth's wall-clock times are in that same offset. If
that assumption is later found to be wrong, only the three `_eastern(...)`
calls below need to change.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

_EASTERN = timezone(timedelta(hours=-4))  # EDT -- see "Time zone note" above


def _canon_host(name: str) -> str:
    """Normalize a hostname for ground-truth matching.

    Upper-cases and strips a trailing ".systemia.com" domain suffix, so
    the real eCAR spelling ("SysClient0358.systemia.com"), and the
    ground-truth PDF's own inconsistent spellings for the same host
    ("Sysclient0358", "SYSCLIENT0358"), and the domain controller's two
    spellings ("DC1" on day 1, "DC1.systemia.com" on day 2), all
    collapse to one canonical token ("SYSCLIENT0358", "DC1").
    """
    n = name.strip().upper()
    if n.endswith(".SYSTEMIA.COM"):
        n = n[: -len(".SYSTEMIA.COM")]
    return n


def _eastern(year: int, month: int, day: int, hour: int, minute: int, second: int) -> float:
    return datetime(year, month, day, hour, minute, second, tzinfo=_EASTERN).timestamp()


@dataclasses.dataclass(frozen=True)
class RedTeamDay:
    label: str
    hosts: frozenset
    start: float  # UTC epoch seconds
    end: float  # UTC epoch seconds


_DAY1_HOSTS = frozenset(
    _canon_host(h)
    for h in [
        "Sysclient0201", "Sysclient0402", "Sysclient0660", "DC1",
        "SYSCLIENT0104", "SYSCLIENT0170", "SYSCLIENT0205", "SYSCLIENT0255",
        "SYSCLIENT0321", "SYSCLIENT0355", "SYSCLIENT0419", "SYSCLIENT0462",
        "SYSCLIENT0503", "SYSCLIENT0559", "SYSCLIENT0609", "SYSCLIENT0771",
        "SYSCLIENT0874", "SYSCLIENT0955",
    ]
)

_DAY2_HOSTS = frozenset(
    _canon_host(h)
    for h in [
        "Sysclient0501", "Sysclient0811", "DC1.systemia.com",
        "Sysclient0974", "Sysclient0005", "Sysclient0010", "Sysclient0069",
        "Sysclient0203", "Sysclient0358", "Sysclient0618", "Sysclient0851",
    ]
)

_DAY3_HOSTS = frozenset(_canon_host(h) for h in ["Sysclient0051", "Sysclient0351"])


RED_TEAM_DAYS: list[RedTeamDay] = [
    RedTeamDay(
        label="day1_plain_powershell_empire",
        hosts=_DAY1_HOSTS,
        start=_eastern(2019, 9, 23, 11, 23, 29),
        end=_eastern(2019, 9, 23, 15, 30, 0),
    ),
    RedTeamDay(
        label="day2_custom_powershell_empire",
        hosts=_DAY2_HOSTS,
        start=_eastern(2019, 9, 24, 10, 28, 56),
        end=_eastern(2019, 9, 25, 10, 0, 0),
    ),
    RedTeamDay(
        label="day3_malicious_upgrade",
        hosts=_DAY3_HOSTS,
        start=_eastern(2019, 9, 25, 10, 29, 42),
        end=_eastern(2019, 9, 25, 14, 24, 3),
    ),
]

#: Union of every compromised host across all three days, handy for
#: quick membership checks that do not need the day/time breakdown.
ALL_COMPROMISED_HOSTS: frozenset = frozenset().union(*(d.hosts for d in RED_TEAM_DAYS))


def is_malicious(hostname_a: str, hostname_b: str, epoch_t: float) -> bool:
    """True if a FLOW event between `hostname_a` and `hostname_b` at
    absolute UTC time `epoch_t` (seconds since the Unix epoch) falls
    inside a red-team engagement window where BOTH hosts are on that
    day's compromised list.

    `hostname_a` / `hostname_b` may be given in any of the spellings
    documented in this module's docstring (mixed case, with or without
    the ".systemia.com" suffix) -- both are canonicalized before
    matching. Passing a raw IP address (as happens when a FLOW's far
    endpoint does not resolve to a known host) simply never matches any
    canonical hostname and returns False, which is the safe default.

    See the module docstring's "Known limitation" for why this is a
    host+day join, not an event-level one, and what that means for
    interpreting the result.
    """
    a = _canon_host(hostname_a)
    b = _canon_host(hostname_b)
    for day in RED_TEAM_DAYS:
        if day.start <= epoch_t <= day.end and a in day.hosts and b in day.hosts:
            return True
    return False
