"""
Groups a time-ordered stream of `Event`s into fixed-length windows and
builds the per-window graph that `lmsafe.detectors` scores.

A "graph" here is deliberately a plain, dependency-light structure (a node
index plus a weighted directed edge list) rather than a specific library's
graph object -- `lmsafe.detectors.count_baseline` needs almost nothing, and
`lmsafe.detectors.rw_gnn` builds its own `torch` tensors from the same
`WindowGraph`, so neither detector forces a graph-library dependency on the
other.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .schema import Event


@dataclass
class WindowGraph:
    """The host-level interaction graph for one time window.

    Attributes
    ----------
    window_start : int
        Start time of the window, in the same clock as `Event.t`.
    window_seconds : int
        Window length.
    nodes : list[str]
        Sorted, de-duplicated list of every host seen in this window
        (union of src_comp and dst_comp). `node_index[h]` gives a host's
        position in this list.
    node_index : dict[str, int]
        host -> index into `nodes`.
    edges : list[tuple[int, int]]
        Directed (src_index, dst_index) pairs, one entry per *distinct*
        edge (src_comp, dst_comp) seen in the window -- repeats are
        folded into `edge_weight`, not kept as duplicate rows.
    edge_weight : list[int]
        Event count for `edges[i]`, aligned by index.
    edge_is_malicious : list[bool]
        True if *any* event contributing to `edges[i]` was labeled
        malicious. Evaluation-only (see schema.py docstring) -- detectors
        must never read this field.
    n_events : int
        Total event count in the window (== sum(edge_weight)).
    """

    window_start: int
    window_seconds: int
    nodes: list[str] = field(default_factory=list)
    node_index: dict[str, int] = field(default_factory=dict)
    edges: list[tuple[int, int]] = field(default_factory=list)
    edge_weight: list[int] = field(default_factory=list)
    edge_is_malicious: list[bool] = field(default_factory=list)
    n_events: int = 0

    def host_pair(self, i: int) -> tuple[str, str]:
        """Convenience: the (src_host, dst_host) string pair for edges[i]."""
        a, b = self.edges[i]
        return self.nodes[a], self.nodes[b]

    def is_empty(self) -> bool:
        return self.n_events == 0


def filter_edges(window: WindowGraph, keep_indices: Iterable[int]) -> WindowGraph:
    """Return a new `WindowGraph` containing only the edges at
    `keep_indices` (and only the nodes those edges touch).

    Used by `lmsafe.policies.GatedPolicy` to build the "admitted and
    corroborated" sub-graph it is willing to train on, without
    mutating the original window (which the runner may still need
    intact for evaluation).
    """
    keep = list(keep_indices)
    old_to_new: dict[int, int] = {}
    new_nodes: list[str] = []
    new_edges: list[tuple[int, int]] = []
    new_weight: list[int] = []
    new_mal: list[bool] = []

    def remap(old_idx: int) -> int:
        new_idx = old_to_new.get(old_idx)
        if new_idx is None:
            new_idx = len(new_nodes)
            old_to_new[old_idx] = new_idx
            new_nodes.append(window.nodes[old_idx])
        return new_idx

    n_events = 0
    for i in keep:
        a, b = window.edges[i]
        new_edges.append((remap(a), remap(b)))
        new_weight.append(window.edge_weight[i])
        new_mal.append(window.edge_is_malicious[i])
        n_events += window.edge_weight[i]

    return WindowGraph(
        window_start=window.window_start,
        window_seconds=window.window_seconds,
        nodes=new_nodes,
        node_index={h: i for i, h in enumerate(new_nodes)},
        edges=new_edges,
        edge_weight=new_weight,
        edge_is_malicious=new_mal,
        n_events=n_events,
    )


def iter_windows(
    events: Iterable[Event],
    window_seconds: int,
    *,
    align_start: int = 0,
) -> Iterator[WindowGraph]:
    """Consume a time-sorted stream of events and yield one `WindowGraph`
    per non-empty window, in increasing time order.

    A window with zero events is skipped entirely (not yielded as an
    empty graph) -- `lmsafe.policies` and `lmsafe.runner` treat "no
    window at time T" and "an empty window at time T" as the same
    situation, and skipping avoids every caller having to special-case
    `WindowGraph.is_empty()`.

    Parameters
    ----------
    events : Iterable[Event]
        Must be sorted by `.t` ascending. This holds for every reader in
        `lmsafe.data` and for `lmsafe.synth.generate`; it is *not*
        re-validated here (an assert on every event would be a real cost
        on a 1e8-event stream), so a caller feeding unsorted events will
        get wrong, not loudly-erroring, windows. See
        tests/test_windows.py for a test that an out-of-order input
        raises in debug-friendly test code paths.
    window_seconds : int
        Window length in seconds.
    align_start : int
        Window boundaries are `align_start + k * window_seconds` for
        integer k -- i.e. windows are anchored to this value, not to the
        first event's own timestamp. Keeping this fixed across training
        and test extraction is what makes window boundaries comparable
        across a dataset split.
    """
    current_start: int | None = None
    builder: _GraphBuilder | None = None

    for ev in events:
        bucket = align_start + ((ev.t - align_start) // window_seconds) * window_seconds
        if current_start is None:
            current_start = bucket
            builder = _GraphBuilder(current_start, window_seconds)
        elif bucket != current_start:
            if builder is not None and not builder.is_empty():
                yield builder.build()
            current_start = bucket
            builder = _GraphBuilder(current_start, window_seconds)
        builder.add(ev)  # type: ignore[union-attr]

    if builder is not None and not builder.is_empty():
        yield builder.build()


class _GraphBuilder:
    """Accumulates events for one window before `iter_windows` emits it.

    Kept as a private helper (not `WindowGraph` itself) so `WindowGraph`
    can stay a plain, easily-pickled/serialized data holder while the
    O(1)-amortized "have I seen this edge yet" bookkeeping lives here.
    """

    __slots__ = (
        "window_start",
        "window_seconds",
        "_node_index",
        "_edge_index",
        "_edges",
        "_weight",
        "_malicious",
        "_n_events",
    )

    def __init__(self, window_start: int, window_seconds: int):
        self.window_start = window_start
        self.window_seconds = window_seconds
        self._node_index: dict[str, int] = {}
        self._edge_index: dict[tuple[int, int], int] = {}
        self._edges: list[tuple[int, int]] = []
        self._weight: list[int] = []
        self._malicious: list[bool] = []
        self._n_events = 0

    def _node(self, host: str) -> int:
        idx = self._node_index.get(host)
        if idx is None:
            idx = len(self._node_index)
            self._node_index[host] = idx
        return idx

    def add(self, ev: Event) -> None:
        a = self._node(ev.src_comp)
        b = self._node(ev.dst_comp)
        key = (a, b)
        idx = self._edge_index.get(key)
        if idx is None:
            idx = len(self._edges)
            self._edge_index[key] = idx
            self._edges.append(key)
            self._weight.append(0)
            self._malicious.append(False)
        self._weight[idx] += 1
        if ev.is_malicious:
            self._malicious[idx] = True
        self._n_events += 1

    def is_empty(self) -> bool:
        return self._n_events == 0

    def build(self) -> WindowGraph:
        # nodes sorted for determinism (two runs over the same events
        # must produce identical node orderings, independent of hashing).
        ordered_hosts = sorted(self._node_index, key=self._node_index.get)
        return WindowGraph(
            window_start=self.window_start,
            window_seconds=self.window_seconds,
            nodes=ordered_hosts,
            node_index=dict(self._node_index),
            edges=list(self._edges),
            edge_weight=list(self._weight),
            edge_is_malicious=list(self._malicious),
            n_events=self._n_events,
        )
