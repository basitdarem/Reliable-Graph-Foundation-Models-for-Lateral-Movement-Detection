"""Tests for lmsafe.data.optc: the real OpTC FLOW-event reader.

The schema asserted here is not guessed -- it is confirmed against a real
241-line sample of ecar-bro/short (2019-09-25, one host-range partition)
obtained from the official OpTC Google Drive release and inspected
line-by-line. That real sample happens to contain only outbound
connections to a small set of *external* IPs (benign cert-check
beaconing, plus one Notepad++-update-hijack/C2 sequence -- no internal-
to-internal traffic at all), so it alone cannot exercise the
internal-to-internal host resolution path this reader's whole design
rests on. The tests below cover that path with small, realistically-
shaped synthetic FLOW lines (same field names/types as the real sample)
in addition to replaying real sample lines verbatim for the parts real
data *does* cover.
"""
import gzip
import json
from pathlib import Path

from lmsafe.data import optc
from lmsafe.data.optc import OptcConfig, _parse_optc_timestamp, build_ip_host_table, stream_optc_events

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_SAMPLE_FILE = FIXTURES_DIR / "optc_ecarbro_short_sample.json"


def _flow_line(
    hostname,
    direction,
    src_ip,
    dest_ip,
    dest_port="80",
    principal="SYSTEMIACOM\\Administrator",
    timestamp="2019-09-25T09:05:14.574-04:00",
):
    """Build one realistically-shaped FLOW JSON line, matching the real
    sample's confirmed field names exactly."""
    return json.dumps(
        {
            "timestamp": timestamp,
            "id": "98d29eef-5d3f-409c-b326-0cea2df109e7",
            "hostname": hostname,
            "objectID": "49e8ff2d-0c89-494e-92cc-b67981b7333e",
            "object": "FLOW",
            "action": "INFO",
            "actorID": "f247d0d1-6ce4-4f89-aab8-7ebd7fba2ce6",
            "pid": 2984,
            "ppid": 2820,
            "tid": -1,
            "principal": principal,
            "properties": {
                "acuity_level": "1",
                "bro_uid": "CxSHfc16kZy97tnun6",
                "dest_ip": dest_ip,
                "dest_port": dest_port,
                "direction": direction,
                "image_path": "powershell.exe",
                "l4protocol": "6",
                "src_ip": src_ip,
                "src_port": "55398",
            },
        }
    )


# ---------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------


def test_parse_optc_timestamp_handles_variable_fraction_digits():
    """Real OpTC timestamps carry 0, 2, or 3+ fractional digits (all
    three shapes confirmed in the real sample) -- the hand-rolled parser
    exists specifically because `datetime.fromisoformat` is not reliably
    lenient about this across Python versions."""
    no_frac = _parse_optc_timestamp("2019-09-25T09:56:52-04:00")
    two_digit = _parse_optc_timestamp("2019-09-25T09:07:45.26-04:00")
    three_digit = _parse_optc_timestamp("2019-09-25T09:05:14.574-04:00")
    assert isinstance(no_frac, float) and isinstance(two_digit, float) and isinstance(three_digit, float)

    # ordering within one real, verbatim sequence from the sample must survive parsing
    t1 = _parse_optc_timestamp("2019-09-25T09:07:15.106-04:00")
    t2 = _parse_optc_timestamp("2019-09-25T09:07:45.26-04:00")
    t3 = _parse_optc_timestamp("2019-09-25T09:08:15.386-04:00")
    assert t1 < t2 < t3


def test_parse_optc_timestamp_rejects_garbage():
    import pytest

    with pytest.raises(ValueError):
        _parse_optc_timestamp("not a timestamp")


# ---------------------------------------------------------------------
# Real sample lines, replayed verbatim
# ---------------------------------------------------------------------

# Two consecutive lines taken verbatim from the real ecar-bro/short sample.
_REAL_LINE_1 = (
    '{"timestamp":"2019-09-25T09:05:14.574-04:00","id":"98d29eef-5d3f-409c-b326-0cea2df109e7",'
    '"hostname":"SysClient0358.systemia.com","objectID":"49e8ff2d-0c89-494e-92cc-b67981b7333e",'
    '"object":"FLOW","action":"INFO","actorID":"f247d0d1-6ce4-4f89-aab8-7ebd7fba2ce6","pid":2984,'
    '"ppid":2820,"tid":-1,"principal":"SYSTEMIACOM\\\\Administrator","properties":{"acuity_level":"1",'
    '"bro_uid":"CxSHfc16kZy97tnun6","dest_ip":"202.6.172.98","dest_port":"443","direction":"outbound",'
    '"image_path":"powershell.exe","l4protocol":"6","src_ip":"142.20.57.103","src_port":"55398"}}'
)
_REAL_LINE_NOT_FLOW = (
    '{"timestamp":"2019-09-25T09:05:14.574-04:00","id":"x","hostname":"SysClient0358.systemia.com",'
    '"objectID":"y","object":"PROCESS","action":"CREATE","actorID":"z","pid":1,"ppid":2,"tid":-1,'
    '"principal":"SYSTEM","properties":{}}'
)


def test_real_sample_line_parses_and_is_dropped_as_external_only(tmp_path):
    """The real sample's dominant pattern -- outbound to an external IP
    with no matching internal host -- must be dropped under the default
    `only_internal_internal=True`, the OpTC-equivalent of LANL's
    same-host/remote-logon fairness filter."""
    p = tmp_path / "sample.json"
    p.write_text(_REAL_LINE_1 + "\n" + _REAL_LINE_NOT_FLOW + "\n")

    records = list(optc._iter_flow_records([p]))
    assert len(records) == 1, "the PROCESS-object line must not be parsed as a FLOW record"
    assert records[0].hostname == "SysClient0358.systemia.com"
    assert records[0].src_ip == "142.20.57.103"
    assert records[0].dest_ip == "202.6.172.98"

    events = stream_optc_events([p], OptcConfig())
    assert events == [], "an outbound flow to an unresolvable (external) IP must be dropped by default"


def test_malformed_json_line_is_skipped_not_raised(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text(_REAL_LINE_1 + "\n{not valid json\n")
    records = list(optc._iter_flow_records([p]))
    assert len(records) == 1


# ---------------------------------------------------------------------
# Internal-to-internal resolution (the path real data didn't exercise)
# ---------------------------------------------------------------------


def test_ip_host_table_built_purely_from_self_observation():
    lines = [
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "8.8.8.8"),
        _flow_line("SysClient0150.systemia.com", "inbound", "203.0.113.5", "142.20.57.50"),
    ]
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        p = _P(d) / "s.json"
        p.write_text("\n".join(lines) + "\n")
        records = list(optc._iter_flow_records([p]))
    table = build_ip_host_table(records)
    assert table["142.20.57.10"] == "SysClient0100.systemia.com"
    assert table["142.20.57.50"] == "SysClient0150.systemia.com"
    # the external IPs (8.8.8.8, 203.0.113.5) must never appear: there is
    # no self-observation that would attribute them to any host.
    assert "8.8.8.8" not in table
    assert "203.0.113.5" not in table


def test_outbound_internal_to_internal_flow_produces_correct_event(tmp_path):
    """SysClient0100 -> SysClient0150 on port 445 (SMB, a classic
    lateral-movement-relevant port) must become one Event with the
    source/destination hosts in the right order."""
    lines = [
        # self-observation: pins 142.20.57.50 to SysClient0150
        _flow_line("SysClient0150.systemia.com", "outbound", "142.20.57.50", "8.8.8.8", dest_port="80"),
        # the flow under test
        _flow_line(
            "SysClient0100.systemia.com", "outbound", "142.20.57.10", "142.20.57.50",
            dest_port="445", principal="SYSTEMIACOM\\attacker",
        ),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    events = stream_optc_events([p], OptcConfig())
    assert len(events) == 1
    e = events[0]
    assert e.src_comp == "SysClient0100.systemia.com"
    assert e.dst_comp == "SysClient0150.systemia.com"
    assert e.auth_type == "445"
    assert e.logon_type == "outbound"
    assert e.orientation == "LogOn"
    assert e.success is True
    assert e.is_malicious is False
    assert e.source == "optc"
    assert e.src_user == "SYSTEMIACOM\\attacker"
    assert e.dst_user == e.src_user
    assert e.is_remote_logon(), "an emitted OpTC event must satisfy the shared is_remote_logon() check unchanged"


def test_ground_truth_label_defaults_off_and_opts_in(tmp_path):
    """Two real red-team hosts (Sysclient0201 -> Sysclient0402, day 1)
    talking during day 1's documented action window: `is_malicious` must
    stay False unless `label_with_ground_truth=True` is explicitly set,
    and become True when it is."""
    lines = [
        # self-observation: pins 142.20.57.147 to SysClient0402
        _flow_line(
            "SysClient0402.systemia.com", "outbound", "142.20.57.147", "8.8.8.8",
            dest_port="80", timestamp="2019-09-23T13:00:00-04:00",
        ),
        # the flow under test -- inside day 1's 11:23:29-15:30:00 window
        _flow_line(
            "SysClient0201.systemia.com", "outbound", "142.20.56.202", "142.20.57.147",
            dest_port="445", timestamp="2019-09-23T13:30:00-04:00",
        ),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    default_events = stream_optc_events([p], OptcConfig())
    assert len(default_events) == 1
    assert default_events[0].is_malicious is False, "label_with_ground_truth defaults to False"

    labeled_events = stream_optc_events([p], OptcConfig(label_with_ground_truth=True))
    assert len(labeled_events) == 1
    assert labeled_events[0].is_malicious is True


def test_ground_truth_label_false_outside_engagement_window(tmp_path):
    """Same two red-team hosts, but a timestamp well outside any
    documented engagement window: must not be labeled malicious even
    with `label_with_ground_truth=True`."""
    lines = [
        _flow_line(
            "SysClient0402.systemia.com", "outbound", "142.20.57.147", "8.8.8.8",
            dest_port="80", timestamp="2019-09-20T09:00:00-04:00",
        ),
        _flow_line(
            "SysClient0201.systemia.com", "outbound", "142.20.56.202", "142.20.57.147",
            dest_port="445", timestamp="2019-09-20T09:05:00-04:00",
        ),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    events = stream_optc_events([p], OptcConfig(label_with_ground_truth=True))
    assert len(events) == 1
    assert events[0].is_malicious is False


def test_inbound_internal_to_internal_flow_swaps_src_dst_correctly(tmp_path):
    """When the *reporting* host's record has direction="inbound", that
    host is the destination, not the source -- getting this backwards
    would silently reverse every lateral-movement edge's direction."""
    lines = [
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "8.8.8.8"),
        # SysClient0150 reports *receiving* a connection from 142.20.57.10
        _flow_line("SysClient0150.systemia.com", "inbound", "142.20.57.10", "142.20.57.50", dest_port="3389"),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    events = stream_optc_events([p], OptcConfig())
    assert len(events) == 1
    e = events[0]
    assert e.src_comp == "SysClient0100.systemia.com"
    assert e.dst_comp == "SysClient0150.systemia.com"
    assert e.auth_type == "3389"


def test_self_loop_is_dropped():
    """A host whose resolved source and destination are the same host
    (e.g. loopback-ish logging artifacts) is not a lateral-movement edge
    and must not be emitted."""
    import tempfile
    from pathlib import Path as _P

    lines = [
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "142.20.57.10"),
    ]
    with tempfile.TemporaryDirectory() as d:
        p = _P(d) / "s.json"
        p.write_text("\n".join(lines) + "\n")
        events = stream_optc_events([p], OptcConfig())
    assert events == []


def test_only_internal_internal_false_keeps_raw_ip_as_pseudo_host(tmp_path):
    lines = [
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "203.0.113.9", dest_port="443"),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    events = stream_optc_events([p], OptcConfig(only_internal_internal=False))
    assert len(events) == 1
    assert events[0].src_comp == "SysClient0100.systemia.com"
    assert events[0].dst_comp == "203.0.113.9"


def test_t_is_relative_to_earliest_event_in_the_batch(tmp_path):
    lines = [
        _flow_line(
            "SysClient0150.systemia.com", "outbound", "142.20.57.50", "8.8.8.8",
            timestamp="2019-09-25T09:00:00-04:00",
        ),
        _flow_line(
            "SysClient0100.systemia.com", "outbound", "142.20.57.10", "142.20.57.50", dest_port="445",
            timestamp="2019-09-25T09:00:30-04:00",
        ),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    events = stream_optc_events([p], OptcConfig())
    assert len(events) == 1
    assert events[0].t == 30


# ---------------------------------------------------------------------
# gzip auto-detection
# ---------------------------------------------------------------------


def test_gzip_compressed_file_is_read_transparently(tmp_path):
    lines = [
        _flow_line("SysClient0150.systemia.com", "outbound", "142.20.57.50", "8.8.8.8"),
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "142.20.57.50", dest_port="445"),
    ]
    p = tmp_path / "sample.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    events = stream_optc_events([p], OptcConfig())
    assert len(events) == 1
    assert events[0].src_comp == "SysClient0100.systemia.com"


def test_empty_file_list_returns_empty():
    assert stream_optc_events([], OptcConfig()) == []


# ---------------------------------------------------------------------
# The actual real sample file bundled with this project (not a single
# line copy-pasted into this test file, but the full 241-line file the
# user obtained from their own ecar-bro/short Drive folder and that
# lmsafe_cache/README.md's "Real OpTC support" section and the Colab
# notebook both now run the real reader against end to end).
# ---------------------------------------------------------------------


def test_real_bundled_sample_file_is_parsed_and_yields_zero_internal_flows():
    """Pins down, against the actual bundled file (not a synthetic
    stand-in), the honest finding this reader was built against: all
    240 FLOW records in this specific ecar-bro/short partition are
    outbound to external IPs, so the default (fair) filter correctly
    yields zero events. If this ever starts returning non-zero events,
    either the file changed or the filter regressed -- both worth
    knowing immediately."""
    assert REAL_SAMPLE_FILE.exists(), "the real OpTC sample fixture is missing"

    records = list(optc._iter_flow_records([REAL_SAMPLE_FILE]))
    assert len(records) == 240

    events_default = stream_optc_events([REAL_SAMPLE_FILE], OptcConfig())
    assert events_default == [], "this real sample is known to contain zero internal-to-internal flows"

    events_keep_external = stream_optc_events(
        [REAL_SAMPLE_FILE], OptcConfig(only_internal_internal=False)
    )
    assert len(events_keep_external) == 240, "with the fairness filter off, every FLOW record becomes an event"
    assert all(e.source == "optc" and e.orientation == "LogOn" for e in events_keep_external)


# ---------------------------------------------------------------------
# Integration with the dataset-agnostic windows/split pipeline
# ---------------------------------------------------------------------


def test_events_feed_iter_windows_and_split_train_test_unchanged(tmp_path):
    """The whole point of the Event boundary: nothing downstream needs
    to know these events came from OpTC. Build a few internal-internal
    flows spread across more than one window and confirm `iter_windows`
    and `lanl.split_train_test` (duck-typed on `cfg.train_cutoff_s`)
    both work on OpTC events exactly as they do on LANL/synthetic ones.
    """
    from lmsafe.data import lanl
    from lmsafe.data.windows import iter_windows

    lines = [
        _flow_line("SysClient0150.systemia.com", "outbound", "142.20.57.50", "8.8.8.8",
                   timestamp="2019-09-25T09:00:00-04:00"),
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "142.20.57.50", dest_port="445",
                   timestamp="2019-09-25T09:00:30-04:00"),
        _flow_line("SysClient0100.systemia.com", "outbound", "142.20.57.10", "142.20.57.50", dest_port="445",
                   timestamp="2019-09-25T11:00:30-04:00"),
    ]
    p = tmp_path / "sample.json"
    p.write_text("\n".join(lines) + "\n")

    cfg = OptcConfig(window_seconds=3600, train_cutoff_s=1800)
    events = stream_optc_events([p], cfg)
    assert len(events) == 2

    train, test_iter = lanl.split_train_test(iter(events), cfg)
    assert len(train) == 1, "only the t=30 event is before the 1800s cutoff"
    test = list(test_iter)
    assert len(test) == 1

    windows = list(iter_windows(events, cfg.window_seconds))
    assert len(windows) == 2, "the two flows are two hours apart -- different 1-hour windows"
    assert windows[0].nodes == ["SysClient0100.systemia.com", "SysClient0150.systemia.com"]
