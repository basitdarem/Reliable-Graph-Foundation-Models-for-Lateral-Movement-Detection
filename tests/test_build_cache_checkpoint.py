"""Regression tests for scripts/fetch_lanl_window.py's `build-cache`
checkpointing.

Context: a real 58-day LANL run was interrupted (Ctrl+C / a Colab
disconnect) at ~87% progress after several hours, and the user found
*nothing* written to disk -- the old `cmd_build_cache` only pickled
`(train_w, test_w)` once, at the very end, after the entire multi-hour
stream had been fully consumed. These tests exercise the fix: periodic
atomic checkpoints during the test-window loop, so an interruption
partway through still leaves a usable, loadable cache file.

`scripts/` has no `__init__.py` (it's a collection of CLI entry points,
not an importable package), so the module is loaded directly from its
file path rather than via a dotted import.
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import pytest

from lmsafe.data.windows import WindowGraph

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "fetch_lanl_window.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_lanl_window", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fetch_lanl_window = _load_module()


def _window(i: int) -> WindowGraph:
    """A trivial, distinct, non-empty window so successive checkpoints
    are easy to tell apart by length alone."""
    w = WindowGraph(window_start=i * 3600, window_seconds=3600)
    w.nodes = [f"C{i}"]
    w.node_index = {f"C{i}": 0}
    w.n_events = 1
    return w


def test_atomic_pickle_dump_leaves_no_tmp_file(tmp_path):
    out = tmp_path / "cache.pkl"
    fetch_lanl_window._atomic_pickle_dump(([1, 2], [3, 4]), out)
    assert out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()
    with open(out, "rb") as fh:
        assert pickle.load(fh) == ([1, 2], [3, 4])


def test_atomic_pickle_dump_overwrite_keeps_latest(tmp_path):
    out = tmp_path / "cache.pkl"
    fetch_lanl_window._atomic_pickle_dump(([], ["a"]), out)
    fetch_lanl_window._atomic_pickle_dump(([], ["a", "b", "c"]), out)
    with open(out, "rb") as fh:
        train, test = pickle.load(fh)
    assert test == ["a", "b", "c"]
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_build_cache_writes_checkpoint_before_completion(tmp_path, monkeypatch):
    """The core regression guard: simulate a `build-cache` run that gets
    interrupted partway through the test-window stream (the real-world
    failure mode -- a long-running Colab cell killed by Ctrl+C or a
    disconnect), and confirm a *usable* checkpoint was already on disk
    before the interruption, covering most but not all of the windows.
    """

    class _Interrupted(Exception):
        pass

    total_windows = 50
    checkpoint_every = 10
    kill_after = 37  # strictly between two checkpoint boundaries (30, 40)

    # `cmd_build_cache` calls `iter_windows` twice: once to build the
    # (small, eager) train-window list, once for the (long, streamed)
    # test-window list. Only the *second* call is the one a real
    # multi-hour run would be interrupted during -- the first call must
    # complete normally so the test exercises the intended code path.
    calls = {"n": 0}

    def fake_iter_windows(events_iter, window_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _window(0)  # trivial, already-complete train pass
            return
        for i in range(total_windows):
            if i == kill_after:
                raise _Interrupted("simulated Ctrl+C / Colab disconnect")
            yield _window(i)

    monkeypatch.setattr(fetch_lanl_window, "iter_windows", fake_iter_windows)
    monkeypatch.setattr(
        fetch_lanl_window.lanl_stream, "resolve_base_url", lambda: "https://example.invalid/cyber1/"
    )
    monkeypatch.setattr(fetch_lanl_window.lanl, "load_redteam_records", lambda base_url: [])
    monkeypatch.setattr(
        fetch_lanl_window.lanl,
        "stream_auth_events",
        lambda cfg, base_url, redteam: iter(()),
    )
    monkeypatch.setattr(
        fetch_lanl_window.lanl, "split_train_test", lambda events_iter, cfg: ([], iter(()))
    )

    out = tmp_path / "lanl_dev_windows.pkl"
    args = type(
        "A",
        (),
        dict(
            end_s=100,
            start_s=0,
            train_cutoff_s=0,
            no_logon_filter=False,
            out=str(out),
            checkpoint_every=checkpoint_every,
        ),
    )()

    with pytest.raises(_Interrupted):
        fetch_lanl_window.cmd_build_cache(args)

    # The run never reached the end (window 50), so no *final* write ever
    # happened -- but it must still have left the last periodic
    # checkpoint (at window 30, the last multiple of 10 before the kill
    # at 37) on disk, loadable and usable right away.
    assert out.exists(), "interrupted run left nothing on disk -- checkpointing regressed"
    with open(out, "rb") as fh:
        train_w, test_w = pickle.load(fh)
    assert len(test_w) == 30, (
        f"expected the last checkpoint before the kill (30 windows), got {len(test_w)}"
    )
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_build_cache_final_file_has_all_windows_on_clean_completion(tmp_path, monkeypatch):
    total_windows = 23  # deliberately not a multiple of checkpoint_every

    calls = {"n": 0}

    def fake_iter_windows(events_iter, window_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _window(0)  # trivial train pass
            return
        for i in range(total_windows):
            yield _window(i)

    monkeypatch.setattr(fetch_lanl_window, "iter_windows", fake_iter_windows)
    monkeypatch.setattr(
        fetch_lanl_window.lanl_stream, "resolve_base_url", lambda: "https://example.invalid/cyber1/"
    )
    monkeypatch.setattr(fetch_lanl_window.lanl, "load_redteam_records", lambda base_url: [])
    monkeypatch.setattr(
        fetch_lanl_window.lanl,
        "stream_auth_events",
        lambda cfg, base_url, redteam: iter(()),
    )
    monkeypatch.setattr(
        fetch_lanl_window.lanl, "split_train_test", lambda events_iter, cfg: ([], iter(()))
    )

    out = tmp_path / "cache.pkl"
    args = type(
        "A",
        (),
        dict(
            end_s=100,
            start_s=0,
            train_cutoff_s=0,
            no_logon_filter=False,
            out=str(out),
            checkpoint_every=10,
        ),
    )()

    fetch_lanl_window.cmd_build_cache(args)

    with open(out, "rb") as fh:
        train_w, test_w = pickle.load(fh)
    # The *final* write must include every window, not just the last
    # checkpoint boundary (20) -- the trailing 3 windows (20, 21, 22)
    # must not be silently dropped.
    assert len(test_w) == total_windows
    assert not out.with_suffix(out.suffix + ".tmp").exists()
