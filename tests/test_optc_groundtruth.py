"""Tests for lmsafe.data.optc_groundtruth: the OpTC red-team host+day
ground truth transcribed from OpTCRedTeamGroundTruth.pdf.

Values asserted below (hosts, dates, times) are copied directly from
that document -- see the module's own docstring for the full narrative
and the host+day-not-event-level caveat.
"""
from lmsafe.data import optc_groundtruth as gt


def _ts(year, month, day, hour, minute, second):
    return gt._eastern(year, month, day, hour, minute, second)


def test_canon_host_collapses_all_documented_spellings():
    assert gt._canon_host("SysClient0358.systemia.com") == "SYSCLIENT0358"
    assert gt._canon_host("Sysclient0358") == "SYSCLIENT0358"
    assert gt._canon_host("SYSCLIENT0358") == "SYSCLIENT0358"
    assert gt._canon_host("DC1.systemia.com") == "DC1"
    assert gt._canon_host("DC1") == "DC1"


def test_day1_pair_inside_window_is_malicious():
    # Sysclient0201 -> Sysclient0402 is the day-1 pivot documented at
    # 09/23/19 13:24:36 (Eastern); any time within that day's window
    # should match.
    t = _ts(2019, 9, 23, 13, 30, 0)
    assert gt.is_malicious("Sysclient0201.systemia.com", "SysClient0402.systemia.com", t)
    # order must not matter
    assert gt.is_malicious("SysClient0402.systemia.com", "Sysclient0201.systemia.com", t)


def test_day1_pair_outside_window_is_not_malicious():
    before = _ts(2019, 9, 23, 9, 0, 0)
    after = _ts(2019, 9, 23, 16, 0, 0)
    assert not gt.is_malicious("Sysclient0201.systemia.com", "SysClient0402.systemia.com", before)
    assert not gt.is_malicious("Sysclient0201.systemia.com", "SysClient0402.systemia.com", after)


def test_only_one_host_compromised_is_not_malicious():
    t = _ts(2019, 9, 23, 13, 30, 0)
    # SysClient9999 is not on any day's compromised list.
    assert not gt.is_malicious("Sysclient0201.systemia.com", "SysClient9999.systemia.com", t)


def test_dc1_spelling_matches_across_day1_and_day2():
    # Day 1 documents the domain controller as "DC1"; day 2 documents it
    # as "DC1.systemia.com" -- both must resolve to the same canonical host.
    day1_t = _ts(2019, 9, 23, 14, 30, 0)
    day2_t = _ts(2019, 9, 24, 11, 0, 0)
    assert gt.is_malicious("DC1", "SYSCLIENT0104", day1_t)
    assert gt.is_malicious("DC1.systemia.com", "Sysclient0501", day2_t)


def test_day2_overnight_persistence_checkin_is_inside_window():
    # 09/25/19 10:00:00 is the documented end of day 2's window (the
    # overnight WMI-subscription persistence re-check-in); the boundary
    # itself must be inclusive.
    t = _ts(2019, 9, 25, 10, 0, 0)
    assert gt.is_malicious("Sysclient0501", "DC1.systemia.com", t)


def test_day3_pair_inside_window_is_malicious():
    t = _ts(2019, 9, 25, 11, 0, 0)
    assert gt.is_malicious("Sysclient0051", "Sysclient0351", t)


def test_cross_day_pair_is_not_malicious():
    # Sysclient0201 (day 1 only) and Sysclient0501 (day 2 only) were
    # never both compromised at the same time.
    t = _ts(2019, 9, 24, 11, 0, 0)
    assert not gt.is_malicious("Sysclient0201", "Sysclient0501", t)


def test_raw_ip_never_matches_any_host():
    t = _ts(2019, 9, 23, 13, 30, 0)
    assert not gt.is_malicious("142.20.56.202", "SysClient0402.systemia.com", t)


def test_all_compromised_hosts_union_is_nonempty_and_deduplicated():
    assert len(gt.ALL_COMPROMISED_HOSTS) > 0
    # DC1 appears on both day 1 and day 2 lists -- the union must not
    # double count it as two different tokens.
    assert "DC1" in gt.ALL_COMPROMISED_HOSTS
