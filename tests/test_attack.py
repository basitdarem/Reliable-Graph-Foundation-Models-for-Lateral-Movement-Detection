from lmsafe import synth
from lmsafe.attack import ConditioningSpec, boiling_frog_scenario, generate_conditioning_events, inject_conditioning
from lmsafe.data.schema import Event


def make_episode(start_s=20 * 86400):
    return synth.LMEpisode(
        start_s=start_s,
        actor_user="U_attacker@DOM1",
        actor_host="C_attacker",
        target_hosts=["C_t1", "C_t2", "C_t3"],
        hop_seconds=30,
    )


def test_conditioning_events_stay_within_window_and_target_the_actor_pair():
    spec = ConditioningSpec(
        start_s=1000, end_s=1000 + 10 * 86400, actor_user="U@DOM1",
        actor_host="C_actor", target_host="C_target", events_per_day=5.0, seed=0,
    )
    events = generate_conditioning_events(spec)
    assert len(events) > 0
    for e in events:
        assert spec.start_s <= e.t < spec.end_s
        assert e.src_comp == "C_actor"
        assert e.dst_comp == "C_target"
        assert e.source == "synthetic_conditioning"


def test_conditioning_events_not_malicious_by_default():
    spec = ConditioningSpec(
        start_s=0, end_s=5 * 86400, actor_user="U@DOM1", actor_host="C_actor",
        target_host="C_target", events_per_day=5.0, seed=1,
    )
    events = generate_conditioning_events(spec)
    assert events, "expected some events to be generated"
    assert all(not e.is_malicious for e in events)


def test_conditioning_events_can_opt_into_malicious_label():
    spec = ConditioningSpec(
        start_s=0, end_s=5 * 86400, actor_user="U@DOM1", actor_host="C_actor",
        target_host="C_target", events_per_day=5.0, label_as_malicious=True, seed=1,
    )
    events = generate_conditioning_events(spec)
    assert events and all(e.is_malicious for e in events)


def test_empty_window_produces_no_events():
    spec = ConditioningSpec(start_s=100, end_s=100, actor_user="U", actor_host="A", target_host="B")
    assert generate_conditioning_events(spec) == []


def test_boiling_frog_scenario_targets_same_actor_and_first_hop():
    ep = make_episode()
    spec = boiling_frog_scenario(ep, conditioning_days=7, events_per_day=2.0, gap_seconds=3600, seed=0)
    assert spec.actor_host == ep.actor_host
    assert spec.actor_user == ep.actor_user
    assert spec.target_host == ep.target_hosts[0]
    assert spec.end_s == ep.start_s - 3600
    assert spec.start_s == spec.end_s - 7 * 86400


def test_boiling_frog_scenario_never_overlaps_the_real_attack():
    ep = make_episode()
    spec = boiling_frog_scenario(ep, conditioning_days=7, gap_seconds=3600, seed=0)
    assert spec.end_s < ep.start_s


def test_inject_conditioning_merges_and_keeps_stream_sorted():
    base = [
        Event(t=0, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2", source="synthetic"),
        Event(t=10 * 86400, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2", source="synthetic"),
    ]
    spec = ConditioningSpec(
        start_s=1000, end_s=5 * 86400, actor_user="U@DOM1", actor_host="C_a",
        target_host="C_b", events_per_day=10.0, seed=2,
    )
    merged = inject_conditioning(base, spec)
    assert [e.t for e in merged] == sorted(e.t for e in merged)
    assert len(merged) > len(base)
    # original list must not be mutated
    assert len(base) == 2


def test_inject_conditioning_with_no_events_returns_equivalent_list():
    base = [Event(t=0, src_user="a", dst_user="a", src_comp="C1", dst_comp="C2", source="synthetic")]
    spec = ConditioningSpec(start_s=5, end_s=5, actor_user="U", actor_host="A", target_host="B")
    merged = inject_conditioning(base, spec)
    assert merged == base
    assert merged is not base
