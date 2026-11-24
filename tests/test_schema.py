from lmsafe.data.schema import Event, RedTeamRecord, event_from_csv_row, event_to_csv_row


def test_event_edge_and_remote_logon():
    e = Event(t=1, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2", orientation="LogOn")
    assert e.edge == ("C1", "C2")
    assert e.is_remote_logon()


def test_same_host_logon_is_not_remote():
    e = Event(t=1, src_user="a", dst_user="a", src_comp="C1", dst_comp="C1", orientation="LogOn")
    assert not e.is_remote_logon()


def test_logoff_is_not_remote_logon():
    e = Event(t=1, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2", orientation="LogOff")
    assert not e.is_remote_logon()


def test_logon_type_alone_does_not_make_it_a_remote_logon():
    # Regression guard for the exact bug this project shipped: LANL's
    # `logon_type` column holds the Windows auth *mechanism*
    # ("Network", "Batch", "Interactive", ...) and is never literally
    # "LogOn" -- that marker lives in `orientation`. An Event with
    # logon_type="LogOn" but no matching orientation must NOT be
    # treated as a remote logon, and real-shaped logon_type values
    # (e.g. "Network") with the correct orientation must still pass.
    wrong_field = Event(
        t=1, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2",
        logon_type="LogOn", orientation="",
    )
    assert not wrong_field.is_remote_logon()

    real_shaped = Event(
        t=1, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2",
        logon_type="Network", orientation="LogOn",
    )
    assert real_shaped.is_remote_logon()


def test_csv_roundtrip():
    e = Event(
        t=42,
        src_user="U1@DOM1",
        dst_user="U2@DOM1",
        src_comp="C1",
        dst_comp="C2",
        auth_type="Kerberos",
        logon_type="LogOn",
        orientation="LogOn",
        success=True,
        is_malicious=True,
        source="synthetic",
    )
    row = event_to_csv_row(e)
    e2 = event_from_csv_row({k: str(v) for k, v in row.items()})
    assert e2 == e


def test_redteam_record_key_uses_dst_user_not_src_user():
    # Regression guard for the exact join-key bug this project's own
    # literature review flagged as easy to get backwards (see
    # RedTeamRecord.key docstring): the key must be built from the
    # record's own fields in (t, user, src_comp, dst_comp) order, with
    # "user" matched against an Event's dst_user elsewhere, not here.
    r = RedTeamRecord(t=100, user="U1@DOM1", src_comp="C1", dst_comp="C2")
    assert r.key() == (100, "U1@DOM1", "C1", "C2")
