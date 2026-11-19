"""Model-owned alias incidence index and request-local canonical views."""

import pytest

from bilt.core._structural_analysis._causalization_block_solver._alias_matching import (
    alias_matching_scope,
    build_alias_matching,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import inspect_structure

from .test_alias_matching import connect, origin, sink


def test_alias_index_is_reused_for_equivalent_supplied_sets() -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    index = incidence.canonical_index
    view = index.view({"input_value"})
    assert index.view(set()) is view
    assert index.view({"answer"}) is view
    assert view.canonical("a") == view.canonical("b")
    assert len(view.spanning) == 1
    assert len(index.links) == 1


def test_supplied_alias_breaks_cached_partition() -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    index = incidence.canonical_index
    baseline = index.view({"input_value"})
    blocked = index.view({"input_value", "a"})
    assert blocked is not baseline
    assert blocked.canonical("a") != blocked.canonical("b")
    assert index.view({"a"}) is blocked
    assert build_alias_matching(incidence, {"a", "input_value"}) is None


@pytest.mark.parametrize(
    ("available", "target"),
    (
        (frozenset(), "a"),
        (frozenset({"input_value"}), "answer"),
        (frozenset({"a", "input_value"}), "answer"),
        (frozenset({"b", "input_value"}), "answer"),
        (frozenset({"answer"}), "answer"),
    ),
)
def test_indexed_scope_matches_baseline(available: frozenset[str], target: str) -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    ordinary = inspect_structure(incidence, available, {target})
    with alias_matching_scope():
        indexed = inspect_structure(incidence, available, {target})
    assert indexed == ordinary


def test_indexed_matching_preserves_requested_outputs() -> None:
    incidence = FullIncidenceGraph([origin, connect, sink])
    first = build_alias_matching(incidence, {"input_value"})
    second = build_alias_matching(incidence, {"input_value"})
    assert first is not None and second is not None
    assert first.representatives == second.representatives
    assert first.removed_nodes == second.removed_nodes
    assert first.removed_edges == second.removed_edges
