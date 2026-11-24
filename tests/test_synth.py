from lmsafe.config import LANL_TRAIN_CUTOFF_S, SynthConfig
from lmsafe import synth


def small_cfg(**overrides) -> SynthConfig:
    base = dict(
        n_users=15,
        n_hosts=12,
        n_benign_days=6,
        base_edges_per_window=10,
        n_episodes=3,
        episode_chain_len=4,
        seed=1,
    )
    base.update(overrides)
    return SynthConfig(**base)


def test_generate_is_sorted_and_deterministic_for_fixed_seed():
    cfg = small_cfg()
    a = synth.generate(cfg)
    b = synth.generate(cfg)
    assert [e.t for e in a] == sorted(e.t for e in a)
    assert a == b


def test_different_seeds_change_the_stream():
    a = synth.generate(small_cfg(seed=1))
    b = synth.generate(small_cfg(seed=2))
    assert a != b


def test_malicious_event_count_matches_episode_spec():
    cfg = small_cfg(n_episodes=3, episode_chain_len=4)
    events = synth.generate(cfg)
    malicious = [e for e in events if e.is_malicious]
    assert len(malicious) == cfg.n_episodes * cfg.episode_chain_len


def test_malicious_events_have_clean_plain_str_fields():
    # Regression guard: numpy's Generator.choice returns numpy.str_
    # scalars, which must be cast back to plain str (see synth.py
    # comments) so downstream dict keys / CSV writers behave normally.
    events = synth.generate(small_cfg())
    malicious = [e for e in events if e.is_malicious]
    assert malicious, "expected at least one injected episode"
    for e in malicious:
        assert type(e.src_user) is str
        assert type(e.src_comp) is str
        assert type(e.dst_comp) is str


def test_episodes_each_have_one_actor_host_and_distinct_targets():
    cfg = small_cfg(n_episodes=2, episode_chain_len=5)
    events = synth.generate(cfg)
    malicious = [e for e in events if e.is_malicious]
    by_actor = {}
    for e in malicious:
        by_actor.setdefault(e.src_comp, []).append(e.dst_comp)
    assert len(by_actor) == cfg.n_episodes
    for targets in by_actor.values():
        assert len(targets) == cfg.episode_chain_len
        assert len(set(targets)) == cfg.episode_chain_len  # distinct hops


def test_default_episodes_stay_clear_of_training_cutoff():
    # Regression guard for the min_start_s fix: with a short benign
    # period and several episodes, naive evenly-spaced placement used
    # to put the first episode inside the 41-hour training window,
    # silently removing it from the test population.
    cfg = small_cfg(n_benign_days=15, n_episodes=4, episode_chain_len=4)
    events = synth.generate(cfg)
    malicious_times = [e.t for e in events if e.is_malicious]
    assert min(malicious_times) >= LANL_TRAIN_CUTOFF_S


def test_zero_episodes_gives_purely_benign_stream():
    cfg = small_cfg(n_episodes=0)
    events = synth.generate(cfg, episodes=[])
    assert all(not e.is_malicious for e in events)


def test_no_drift_keeps_host_population_fixed():
    cfg = small_cfg(n_episodes=0, drift_strength=0.0, drift_start_day=None)
    events = synth.generate(cfg, episodes=[])
    hosts = {e.src_comp for e in events} | {e.dst_comp for e in events}
    # population should not exceed the configured size (common servers
    # + desktops); allow equality only.
    assert len(hosts) <= cfg.n_hosts


def test_drift_grows_the_host_population():
    no_drift = small_cfg(n_benign_days=20, n_episodes=0, drift_strength=0.0, drift_start_day=None)
    with_drift = small_cfg(n_benign_days=20, n_episodes=0, drift_strength=1.0, drift_start_day=2)
    hosts_a = {e.src_comp for e in synth.generate(no_drift, episodes=[])} | {
        e.dst_comp for e in synth.generate(no_drift, episodes=[])
    }
    hosts_b = {e.src_comp for e in synth.generate(with_drift, episodes=[])} | {
        e.dst_comp for e in synth.generate(with_drift, episodes=[])
    }
    assert len(hosts_b) > len(hosts_a)
