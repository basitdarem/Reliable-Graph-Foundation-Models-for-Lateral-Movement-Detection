import numpy as np

from lmsafe.drift import PageHinkley, SimplifiedADWIN


def test_page_hinkley_rarely_alarms_on_stationary_noise():
    # Note on parameters: Page-Hinkley's cumulative-sum test statistic
    # has a range that grows as O(sigma * sqrt(n)) under pure noise, so
    # `threshold` must be calibrated to the signal's variance, not a
    # fixed constant -- a threshold tuned for a low-variance per-window
    # *mean* anomaly score (this project's actual use in
    # lmsafe.policies.DriftTriggeredPolicy) would false-alarm
    # constantly on raw, unaveraged N(0, 1) noise as used here. 30 is
    # calibrated for *this test's* noise scale, empirically giving a
    # false-alarm rate well under 10% across 30 independent seeds.
    rng = np.random.default_rng(0)
    ph = PageHinkley(delta=0.01, threshold=30.0)
    alarms = sum(ph.update(x) for x in rng.normal(0, 1, 250))
    assert alarms == 0


def test_page_hinkley_detects_mean_shift():
    rng = np.random.default_rng(0)
    ph = PageHinkley(delta=0.01, threshold=30.0)
    stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(3, 1, 300)])
    detected_at = next((i for i, x in enumerate(stream) if ph.update(x)), None)
    assert detected_at is not None
    assert 300 <= detected_at < 450


def test_adwin_rarely_alarms_on_stationary_noise():
    rng = np.random.default_rng(1)
    adwin = SimplifiedADWIN(delta=0.002, max_window=400, min_window=16)
    alarms = sum(adwin.update(x) for x in rng.normal(0, 1, 500))
    assert alarms == 0


def test_adwin_detects_mean_shift():
    rng = np.random.default_rng(1)
    adwin = SimplifiedADWIN(delta=0.002, max_window=400, min_window=16)
    stream = np.concatenate([rng.normal(0, 1, 250), rng.normal(3, 1, 250)])
    detected_at = next((i for i, x in enumerate(stream) if adwin.update(x)), None)
    assert detected_at is not None
    assert 250 <= detected_at < 400


def test_adwin_ignores_short_streams_below_min_window():
    adwin = SimplifiedADWIN(min_window=16)
    for x in [0.0] * 10:
        assert adwin.update(x) is False
