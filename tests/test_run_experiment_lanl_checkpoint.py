"""Regression tests for scripts/run_experiment.py's `build_lanl_windows`
checkpointing against `lanl.cache_path`.

Context: this is the *actual* code path the user hit in the field -- a
58-day real LANL run invoked as `run_experiment.py configs/lanl_full.yaml`
printed `[run_experiment] streaming LANL auth events ...` (this
function's own message), reached ~87% progress over several hours, was
interrupted (Ctrl+C / a Colab disconnect), and left *nothing* at
`cache_path` -- the old code only ever pickled `(train_w, test_w)` once,
after the entire multi-hour stream had been consumed. `scripts/
fetch_lanl_window.py`'s separate `build-cache` CLI got the same class of
fix, but that is a different code path and does not, by itself, cover
`run_experiment.py`'s own independent streaming branch -- hence this
second, dedicated regression test for the path that actually failed.

`scripts/` has no `__init__.py`, so the module is loaded directly from
its file path rather than via a dotted import (matching
test_build_cache_checkpoint.py's approach for the sibling script).
"""
import importlib.util
import pickle
from pathlib import Path

from lmsafe.data.windows import WindowGraph

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_experiment.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_experiment = _load_module()


def _window(i: int) -> WindowGraph:
    w = WindowGraph(window_start=i * 3600, window_seconds=3600)
    w.nodes = [f"C{i}"]
    w.node_index = {f"C{i}": 0}
    w.n_events = 1
    return w


def _patch_common(monkeypatch, iter_windows_fn):
    monkeypatch.setattr(run_experiment, "iter_windows", iter_windows_fn)
    monkeypatch.setattr(
        run_experiment.lanl.lanl_stream, "resolve_base_url", lambda: "https://example.invalid/cyber1/"
    )
    monkeypatch.setattr(run_experiment.lanl, "load_redteam_records", lambda base_url: [])
    monkeypatch.setattr(
        run_experiment.lanl, "stream_auth_events", lambda cfg, base_url, redteam: iter(())
    )
    monkeypatch.setattr(
        run_experiment.lanl, "split_train_test", lambda events_iter, cfg: ([], iter(()))
    )


def test_build_lanl_windows_checkpoints_before_interruption(tmp_path, monkeypatch):
    """The core regression guard, against the exact function and
    `cache_path` mechanism the field failure went through."""

    class _Interrupted(Exception):
        pass

    total_windows = 50
    kill_after = 37  # strictly between checkpoint boundaries 30 and 40
    calls = {"n": 0}

    def fake_iter_windows(events_iter, window_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _window(0)  # the eager train-window pass
            return
        for i in range(total_windows):
            if i == kill_after:
                raise _Interrupted("simulated Ctrl+C / Colab disconnect")
            yield _window(i)

    _patch_common(monkeypatch, fake_iter_windows)

    cache_path = tmp_path / "lanl_full_windows.pkl"
    cfg_dict = {
        "end_s": 5_011_200,
        "train_cutoff_s": 147_600,
        "only_remote_logon": True,
        "cache_path": str(cache_path),
        "checkpoint_every": 10,
    }

    try:
        run_experiment.build_lanl_windows(cfg_dict)
        assert False, "expected the simulated interruption to propagate"
    except _Interrupted:
        pass

    assert cache_path.exists(), (
        "an interrupted real-LANL run left nothing at cache_path -- "
        "checkpointing regressed in run_experiment.py's own streaming path"
    )
    with open(cache_path, "rb") as fh:
        train_w, test_w = pickle.load(fh)
    assert len(test_w) == 30, f"expected the last checkpoint (30 windows), got {len(test_w)}"
    assert not cache_path.with_suffix(cache_path.suffix + ".tmp").exists()


def test_build_lanl_windows_final_file_complete_on_clean_run(tmp_path, monkeypatch):
    total_windows = 23
    calls = {"n": 0}

    def fake_iter_windows(events_iter, window_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _window(0)
            return
        for i in range(total_windows):
            yield _window(i)

    _patch_common(monkeypatch, fake_iter_windows)

    cache_path = tmp_path / "cache.pkl"
    cfg_dict = {
        "end_s": 2_592_000,
        "train_cutoff_s": 147_600,
        "only_remote_logon": True,
        "cache_path": str(cache_path),
        "checkpoint_every": 10,
    }

    train_w, test_w = run_experiment.build_lanl_windows(cfg_dict)
    assert len(test_w) == total_windows
    with open(cache_path, "rb") as fh:
        cached_train, cached_test = pickle.load(fh)
    assert len(cached_test) == total_windows
    assert not cache_path.with_suffix(cache_path.suffix + ".tmp").exists()


def test_loading_a_partial_checkpoint_prints_a_short_coverage_warning(tmp_path, monkeypatch, capsys):
    """The exact scenario reported in the field: a real 58-day run was
    Ctrl+C'd partway through, leaving a valid, loadable checkpoint that
    covers far fewer days than the config's own `end_s` asks for.
    Simply reusing that cache_path on a later invocation must not look
    identical to a finished run in the logs -- it has to say, loudly,
    how much of the requested window it actually covers."""
    cache_path = tmp_path / "lanl_full_windows.pkl"
    # 30 hourly windows starting at t=0 -> covers up to hour 30 (~1.25
    # days), nowhere near the 58 days `end_s` below asks for.
    test_w = [_window(i) for i in range(30)]
    with open(cache_path, "wb") as fh:
        pickle.dump(([_window(0)], test_w), fh)

    cfg_dict = {
        "end_s": 5_011_200,  # 58 days
        "train_cutoff_s": 147_600,
        "only_remote_logon": True,
        "cache_path": str(cache_path),
    }
    train_w, loaded_test_w = run_experiment.build_lanl_windows(cfg_dict)
    assert len(loaded_test_w) == 30

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "partial checkpoint" in out
    assert "1.25" in out  # covered_days
    assert "58.0" in out  # requested_days


def test_loading_a_complete_cache_prints_no_warning(tmp_path, monkeypatch, capsys):
    cache_path = tmp_path / "cache.pkl"
    # end_s below is 30 hours; a cache whose last window reaches hour 30
    # exactly covers the full requested window, so no warning is due.
    test_w = [_window(i) for i in range(30)]
    with open(cache_path, "wb") as fh:
        pickle.dump(([_window(0)], test_w), fh)

    cfg_dict = {
        "end_s": 30 * 3600,
        "train_cutoff_s": 0,
        "only_remote_logon": True,
        "cache_path": str(cache_path),
    }
    run_experiment.build_lanl_windows(cfg_dict)

    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_build_lanl_windows_without_cache_path_still_works(monkeypatch):
    """No `cache_path` means no checkpointing target -- the function
    must fall back to the plain in-memory list build (no pickling at
    all), not crash for lack of somewhere to checkpoint to."""
    total_windows = 5
    calls = {"n": 0}

    def fake_iter_windows(events_iter, window_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            yield _window(0)
            return
        for i in range(total_windows):
            yield _window(i)

    _patch_common(monkeypatch, fake_iter_windows)

    cfg_dict = {
        "end_s": 2_592_000,
        "train_cutoff_s": 147_600,
        "only_remote_logon": True,
    }
    train_w, test_w = run_experiment.build_lanl_windows(cfg_dict)
    assert len(test_w) == total_windows
