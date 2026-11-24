"""Static enforcement of this project's central fairness invariant:

    No `Detector.score()` / `Detector.fit()` and no `UpdatePolicy.observe()`
    / `UpdatePolicy.fit_initial()` implementation may read ground-truth
    labels (`Event.is_malicious` / `WindowGraph.edge_is_malicious`).

See `lmsafe/data/schema.py`'s module docstring and `lmsafe/policies.py`
line ~74 for the documented rationale: `is_malicious` exists purely for
`lmsafe.metrics` to grade detectors/policies after the fact, via the
single sanctioned read in `WindowScores.from_window`
(`lmsafe/metrics.py`). Anything scored or decided *during* the
experiment must be blind to it, or every AUC/AP/Rec@B number in the
paper would be measuring a detector that cheated.

This is checked with `ast`, not a plain substring search, because
`policies.py` and `detectors/base.py` legitimately *mention*
`edge_is_malicious` in their docstrings (to document the very
invariant this test enforces) -- a substring check over the whole
module source would false-positive on those comments. Parsing the AST
and looking only at `Attribute` nodes (`<expr>.is_malicious` /
`<expr>.edge_is_malicious`) ignores docstrings and comments entirely,
so it only fires on actual code that reads the field.
"""
from __future__ import annotations

import ast
import importlib
import inspect

import pytest

FORBIDDEN_ATTRS = {"is_malicious", "edge_is_malicious"}

# Every module that implements (or could implement) a Detector or an
# UpdatePolicy. Extend this list when a new detector/policy module is
# added -- `test_module_list_is_complete` below guards against
# forgetting to.
CHECKED_MODULES = [
    "lmsafe.policies",
    "lmsafe.detectors.base",
    "lmsafe.detectors.count_baseline",
    "lmsafe.detectors.feature_mlp",
    "lmsafe.detectors.rw_gnn",
]


def _attribute_accesses(tree: ast.AST) -> set[str]:
    """All `.attr` names referenced anywhere in actual code (assignments,
    conditionals, calls, ...) -- not inside string literals/docstrings,
    since those never become `ast.Attribute` nodes."""
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


@pytest.mark.parametrize("module_name", CHECKED_MODULES)
def test_module_never_accesses_ground_truth_label(module_name):
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    tree = ast.parse(source)
    accessed = _attribute_accesses(tree)
    leaked = accessed & FORBIDDEN_ATTRS
    assert not leaked, (
        f"{module_name} contains code that reads {leaked} -- "
        "detectors/policies must be blind to ground-truth labels "
        "(see lmsafe/data/schema.py and lmsafe/metrics.py:WindowScores.from_window)"
    )


def test_docstrings_may_still_document_the_invariant_by_name():
    """Sanity check on the test itself: policies.py and detectors/base.py
    are *expected* to mention `edge_is_malicious` in prose (that's how
    the invariant gets documented for future contributors) -- this
    confirms the AST check above tolerates that and isn't accidentally
    passing only because the docstrings were removed."""
    import lmsafe.detectors.base as base
    import lmsafe.policies as policies

    assert "edge_is_malicious" in inspect.getsource(policies)
    assert "edge_is_malicious" in inspect.getsource(base)


def test_module_list_is_complete():
    """Guard against silently forgetting to add a new detector/policy
    module to CHECKED_MODULES: every .py file under lmsafe/detectors/
    (except __init__) must be covered."""
    import pathlib

    import lmsafe.detectors as detectors_pkg

    pkg_dir = pathlib.Path(detectors_pkg.__file__).parent
    found = {
        f"lmsafe.detectors.{p.stem}"
        for p in pkg_dir.glob("*.py")
        if p.stem != "__init__"
    }
    assert found <= set(CHECKED_MODULES), (
        f"detector module(s) {found - set(CHECKED_MODULES)} are not covered by "
        "test_no_label_leak.py -- add them to CHECKED_MODULES"
    )
