"""Tests for the fair LANL extraction protocol (lmsafe.data.lanl),
exercised entirely offline: no network access, using the real
redteam.txt.gz fixture for the label-loading test and hand-built
Events for the filtering/labeling logic."""
import io

from lmsafe.config import LanlConfig
from lmsafe.data import lanl
from lmsafe.data.schema import Event, RedTeamRecord


def make_cfg(**overrides) -> LanlConfig:
    return LanlConfig(**overrides)


def test_load_real_redteam_fixture(redteam_fixture_path):
    records = lanl.load_redteam_records(local_path=redteam_fixture_path)
    assert len(records) == 749
    assert records[0] == RedTeamRecord(t=150_885, user="U620@DOM1", src_comp="C17693", dst_comp="C1003")
    assert records[-1] == RedTeamRecord(t=2_557_047, user="U737@DOM1", src_comp="C19932", dst_comp="C108")
    # The project's own inspection (2026-11-13): no event before 41h.
    assert min(r.t for r in records) >= 41 * 3600


def test_remote_logon_positive_match_is_labeled_malicious():
    cfg = make_cfg()
    keys = {RedTeamRecord(t=100, user="U1@DOM1", src_comp="C1", dst_comp="C2").key()}
    ev = Event(t=100, src_user="X", dst_user="U1@DOM1", src_comp="C1", dst_comp="C2", logon_type="Network", orientation="LogOn")
    out = lanl.apply_fair_protocol(ev, keys, cfg)
    assert out is not None
    assert out.is_malicious


def test_same_host_logon_is_dropped_even_if_it_would_match():
    cfg = make_cfg()
    # Same (t, user, src, dst) as a red-team key, but src==dst -- the
    # fair protocol's remote-LogOn filter must drop it before the
    # label join ever runs.
    keys = {RedTeamRecord(t=100, user="U1@DOM1", src_comp="C3", dst_comp="C3").key()}
    ev = Event(t=100, src_user="X", dst_user="U1@DOM1", src_comp="C3", dst_comp="C3", logon_type="Network", orientation="LogOn")
    assert lanl.apply_fair_protocol(ev, keys, cfg) is None


def test_logoff_event_is_dropped():
    cfg = make_cfg()
    ev = Event(t=1, src_user="a", dst_user="b", src_comp="C1", dst_comp="C2", logon_type="Network", orientation="LogOff")
    assert lanl.apply_fair_protocol(ev, set(), cfg) is None


def test_benign_remote_logon_not_in_redteam_is_kept_unlabeled():
    cfg = make_cfg()
    ev = Event(t=5, src_user="a", dst_user="b", src_comp="C1", dst_comp="C2", logon_type="Network", orientation="LogOn")
    out = lanl.apply_fair_protocol(ev, set(), cfg)
    assert out is not None
    assert not out.is_malicious


def test_ntlm_only_filter_is_off_by_default():
    cfg = make_cfg()
    assert cfg.filter_ntlm_only is False
    ev = Event(t=1, src_user="a", dst_user="b", src_comp="C1", dst_comp="C2", logon_type="Network", orientation="LogOn", auth_type="Kerberos")
    assert lanl.apply_fair_protocol(ev, set(), cfg) is not None


def test_ntlm_only_filter_can_be_opted_into_for_non_fair_comparison():
    cfg = make_cfg(filter_ntlm_only=True)
    kerberos_ev = Event(t=1, src_user="a", dst_user="b", src_comp="C1", dst_comp="C2", logon_type="Network", orientation="LogOn", auth_type="Kerberos")
    ntlm_ev = Event(t=1, src_user="a", dst_user="b", src_comp="C1", dst_comp="C2", logon_type="Network", orientation="LogOn", auth_type="NTLM")
    assert lanl.apply_fair_protocol(kerberos_ev, set(), cfg) is None
    assert lanl.apply_fair_protocol(ntlm_ev, set(), cfg) is not None


def test_real_lanl_sample_line_same_host_is_dropped():
    # The exact example line from LANL's own published auth.txt.gz
    # schema documentation. Columns: t, src_user, dst_user, src_comp,
    # dst_comp, auth_type, logon_type, orientation, success. src_comp
    # == dst_comp here, so it must be dropped despite orientation
    # being "LogOn".
    raw = b"1,C625$@DOM1,U147@DOM1,C625,C625,Negotiate,Batch,LogOn,Success"
    ev = lanl._auth_fields_to_event(raw.split(b","))
    assert ev is not None
    assert ev.logon_type == "Batch"
    assert ev.orientation == "LogOn"
    assert not ev.is_remote_logon()
    assert lanl.apply_fair_protocol(ev, set(), make_cfg()) is None


def test_real_lanl_peek_line_remote_logon_is_kept():
    # A real line from the user's own auth.txt.gz peek output, with
    # logon_type="Network" (never "LogOn") and orientation="LogOn",
    # src_comp != dst_comp. This is the exact shape of row that the
    # logon_type-vs-orientation mix-up bug rejected 100% of the time
    # across both real LANL runs (kept=0 over 1.5B+ lines) -- this test
    # fails under the old `self.logon_type == "LogOn"` check and must
    # keep passing under the fixed `self.orientation == "LogOn"` check.
    raw = b"1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C1250,C586,NTLM,Network,LogOn,Success"
    ev = lanl._auth_fields_to_event(raw.split(b","))
    assert ev is not None
    assert ev.logon_type == "Network"
    assert ev.orientation == "LogOn"
    assert ev.is_remote_logon()
    out = lanl.apply_fair_protocol(ev, set(), make_cfg())
    assert out is not None
    assert not out.is_malicious


def _ev(t):
    return Event(t=t, src_user="a", dst_user="a", src_comp="C1", dst_comp="C1", source="lanl")


def test_split_train_test_boundary_is_strict_less_than():
    cfg = make_cfg(train_cutoff_s=1000)
    events = [_ev(0), _ev(999), _ev(1000), _ev(1001)]
    train, test_iter = lanl.split_train_test(iter(events), cfg)
    test = list(test_iter)
    assert [e.t for e in train] == [0, 999]
    assert [e.t for e in test] == [1000, 1001]


def test_split_train_test_all_train():
    cfg = make_cfg(train_cutoff_s=1000)
    events = [_ev(0), _ev(1)]
    train, test_iter = lanl.split_train_test(iter(events), cfg)
    assert [e.t for e in train] == [0, 1]
    assert list(test_iter) == []


def test_split_train_test_all_test():
    cfg = make_cfg(train_cutoff_s=1000)
    events = [_ev(1000), _ev(2000)]
    train, test_iter = lanl.split_train_test(iter(events), cfg)
    assert train == []
    assert [e.t for e in test_iter] == [1000, 2000]


def test_split_train_test_empty_input():
    cfg = make_cfg(train_cutoff_s=1000)
    train, test_iter = lanl.split_train_test(iter([]), cfg)
    assert train == []
    assert list(test_iter) == []


def _raw_auth_line(t: int, *, remote: bool) -> bytes:
    """One synthetic raw auth.txt.gz line. `remote=False` builds a
    same-host LogOff line, the kind `apply_fair_protocol` always drops
    -- used below to simulate a stream dominated by filtered-out rows,
    like the real file is."""
    if remote:
        return f"{t},a@DOM1,b@DOM1,C1,C2,Kerberos,Network,LogOn,Success".encode()
    return f"{t},a@DOM1,a@DOM1,C1,C1,?,Network,LogOff,Success".encode()


def test_stream_auth_events_progress_fires_on_raw_lines_not_kept_lines(monkeypatch):
    """Regression test for a real bug found while running this against
    the live LANL stream: the progress-print check used to sit after
    several `continue`s that skip most raw lines (same-host / non-LogOn
    events, which are the majority of real auth.txt.gz traffic). When
    the raw line count crossed a `progress_every` boundary on exactly
    one of those filtered lines, that round's print was silently
    skipped rather than merely delayed -- over a real multi-hour stream
    this produced long, unpredictable stretches with no progress output
    at all even though the stream was reading fine, indistinguishable
    from a hang.

    This test builds a stream that is ~90% filtered-out same-host
    LogOffs by design, specifically so that several `progress_every`
    boundaries land on a filtered line, and checks the print still
    fires exactly once per `progress_every` raw lines regardless.
    """
    progress_every = 10
    total_raw_lines = 55  # crosses the boundary 5 times (10,20,...,50)

    lines = []
    for i in range(total_raw_lines):
        t = i + 1
        # Every 10th raw line (indices 9, 19, 29, ...) is exactly the
        # one that will cross a progress_every boundary (counts are
        # 1-indexed inside stream_auth_events) -- make that specific
        # line a filtered-out same-host LogOff, the worst case for the
        # old, buggy placement of the progress check.
        remote = (i + 1) % progress_every != 0
        lines.append(_raw_auth_line(t, remote=remote))

    def fake_iter_lines(url):
        return iter(lines)

    monkeypatch.setattr(lanl.lanl_stream, "iter_lines", fake_iter_lines)

    cfg = make_cfg(end_s=10_000, train_cutoff_s=0)
    stderr = io.StringIO()
    events = list(
        lanl.stream_auth_events(
            cfg,
            base_url="https://example.invalid/cyber1/",
            redteam=[],
            progress_every=progress_every,
            progress_stream=stderr,
        )
    )

    # Every 10th line was deliberately a same-host LogOff -> dropped by
    # apply_fair_protocol, so kept count is less than raw count, which
    # is exactly the condition that used to hide the progress print.
    assert len(events) < total_raw_lines

    progress_lines = [ln for ln in stderr.getvalue().splitlines() if ln.startswith("[lanl] read")]
    assert len(progress_lines) == total_raw_lines // progress_every
    assert "read 10 lines" in progress_lines[0]
    assert "read 50 lines" in progress_lines[-1]


def _slow_path_keeps(raw: list, cfg: LanlConfig) -> bool:
    """The authoritative, decode-first answer: build the real `Event`
    and ask `apply_fair_protocol` whether it survives (ignoring
    red-team labeling, which is orthogonal to the fair-protocol filter
    this is checking)."""
    parsed = lanl._auth_fields_to_event(raw)
    assert parsed is not None
    return lanl.apply_fair_protocol(parsed, set(), cfg) is not None


def test_raw_prefilter_agrees_with_apply_fair_protocol_on_real_shaped_lines():
    # `stream_auth_events`'s performance optimization: skip utf-8 decode
    # + Event construction entirely for a line the fair protocol would
    # discard anyway. This test is the proof-by-example that the fast,
    # decode-free `_raw_line_passes_fair_prefilter` never disagrees with
    # the slow, authoritative, fully-decoded `apply_fair_protocol` path --
    # across remote/same-host, LogOn/LogOff, and NTLM/Kerberos
    # combinations, with both filters on and off.
    cases = [
        # (src_comp, dst_comp, auth_type, orientation)
        (b"C1", b"C2", b"NTLM", b"LogOn"),      # remote LogOn -> kept
        (b"C1", b"C1", b"NTLM", b"LogOn"),      # same-host -> dropped
        (b"C1", b"C2", b"NTLM", b"LogOff"),     # LogOff -> dropped
        (b"C1", b"C2", b"Kerberos", b"LogOn"),  # remote, non-NTLM
        (b"C1", b"C1", b"Kerberos", b"LogOff"), # same-host AND LogOff
    ]
    for src, dst, auth_t, orient in cases:
        raw = [b"100", b"u@D", b"u@D", src, dst, auth_t, b"Network", orient, b"Success"]
        for cfg in (
            make_cfg(),  # only_remote_logon=True (default), filter_ntlm_only=False
            make_cfg(filter_ntlm_only=True),
            make_cfg(only_remote_logon=False),
            make_cfg(only_remote_logon=False, filter_ntlm_only=True),
        ):
            fast = lanl._raw_line_passes_fair_prefilter(raw, cfg)
            slow = _slow_path_keeps(raw, cfg)
            assert fast == slow, (
                f"fast/slow path disagreement for src={src} dst={dst} "
                f"auth_t={auth_t} orient={orient} cfg={cfg}: fast={fast} slow={slow}"
            )


def test_raw_prefilter_never_falsely_rejects_on_invalid_utf8_hostnames():
    # The one theoretical edge case the prefilter's docstring calls out:
    # two *different*, both-invalid-utf8 byte sequences could in
    # principle decode (errors="replace") to the *same* string, making
    # a byte-level "different hosts" check a safe over-approximation of
    # the real, decoded-string check -- never a false reject. Exercise
    # that directly with genuinely invalid UTF-8 bytes in the host
    # fields: the fast path must never say "drop" when the slow path
    # would keep it.
    weird_hosts = [b"\xff", b"\xfe", b"C1", b"\xff\xfe"]
    for src in weird_hosts:
        for dst in weird_hosts:
            raw = [b"100", b"u@D", b"u@D", src, dst, b"NTLM", b"Network", b"LogOn", b"Success"]
            cfg = make_cfg()
            fast = lanl._raw_line_passes_fair_prefilter(raw, cfg)
            slow = _slow_path_keeps(raw, cfg)
            if slow:
                assert fast, (
                    f"fast path falsely rejected a line the slow path would keep: "
                    f"src={src!r} dst={dst!r}"
                )
