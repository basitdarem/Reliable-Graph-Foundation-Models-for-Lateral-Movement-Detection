from lmsafe.data.schema import Event
from lmsafe.data.windows import filter_edges, iter_windows


def _ev(t, src, dst, mal=False):
    return Event(t=t, src_user="u", dst_user="u", src_comp=src, dst_comp=dst, is_malicious=mal, source="x")


def test_iter_windows_groups_and_dedupes_edges():
    events = [
        _ev(0, "C1", "C2"),
        _ev(10, "C1", "C2"),
        _ev(20, "C2", "C3", mal=True),
        _ev(3700, "C3", "C4"),  # next hour
    ]
    windows = list(iter_windows(events, window_seconds=3600))
    assert len(windows) == 2

    w0 = windows[0]
    assert w0.window_start == 0
    assert w0.nodes == ["C1", "C2", "C3"]
    assert w0.edges == [(0, 1), (1, 2)]
    assert w0.edge_weight == [2, 1]
    assert w0.edge_is_malicious == [False, True]
    assert w0.n_events == 3

    w1 = windows[1]
    assert w1.window_start == 3600
    assert w1.nodes == ["C3", "C4"]


def test_empty_windows_are_skipped_not_yielded_empty():
    events = [_ev(0, "C1", "C2"), _ev(7300, "C1", "C2")]  # two windows apart, middle window empty
    windows = list(iter_windows(events, window_seconds=3600))
    assert [w.window_start for w in windows] == [0, 7200]


def test_iter_windows_on_empty_stream():
    assert list(iter_windows([], window_seconds=3600)) == []


def test_align_start_shifts_window_boundaries():
    events = [_ev(50, "C1", "C2"), _ev(150, "C1", "C2")]
    windows = list(iter_windows(events, window_seconds=100, align_start=25))
    # boundaries at 25, 125, 225, ... -> t=50 in [25,125), t=150 in [125,225)
    assert [w.window_start for w in windows] == [25, 125]


def test_filter_edges_keeps_only_selected_edges_and_remaps_nodes():
    events = [_ev(0, "C1", "C2"), _ev(0, "C3", "C4", mal=True)]
    w = next(iter(iter_windows(events, window_seconds=3600)))
    assert w.nodes == ["C1", "C2", "C3", "C4"]
    sub = filter_edges(w, [1])  # keep only the C3->C4 edge
    assert sub.nodes == ["C3", "C4"]
    assert sub.edges == [(0, 1)]
    assert sub.edge_is_malicious == [True]
    assert sub.n_events == 1
    assert sub.window_start == w.window_start


def test_filter_edges_empty_selection():
    events = [_ev(0, "C1", "C2")]
    w = next(iter(iter_windows(events, window_seconds=3600)))
    sub = filter_edges(w, [])
    assert sub.nodes == []
    assert sub.edges == []
    assert sub.n_events == 0
