"""Model-owned alias incidence index and request-local canonical views."""

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._alias_matching import (
    _restore_graph,
    _snapshot_graph,
    alias_matching_scope,
    build_alias_matching,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import inspect_structure


@equation
def origin(input_value: float, a: float) -> bool:
    return a == input_value * 2


@equation
def connect(a: float, b: float) -> bool:
    return a == b


@equation
def sink(b: float, answer: float) -> bool:
    return answer == 3 * b


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
    assert first is not None
    assert second is not None
    assert first.representatives == second.representatives
    assert first.removed_nodes == second.removed_nodes
    assert first.removed_edges == second.removed_edges


def test_in_place_projection_preserves_native_fallback_graph() -> None:
    from bilt.core._structural_analysis._causalization_block_solver._matching import build_matching_graph  # noqa: PLC0415

    incidence = FullIncidenceGraph([origin, connect, sink])
    graph = build_matching_graph(incidence, {"input_value"})
    before = tuple((graph[a], graph[b], graph.get_edge_data(a, b)) for a, b in graph.edge_list())
    snapshot = _snapshot_graph(graph)
    projected = build_alias_matching(incidence, {"input_value"}, matching_graph=graph, in_place=True)
    assert projected is not None
    assert projected.graph is graph
    assert projected.removed_nodes > 0
    restored = _restore_graph(snapshot)
    after = tuple((restored[a], restored[b], restored.get_edge_data(a, b)) for a, b in restored.edge_list())
    assert after == before


@system
class SharedPorts(metaclass=Interface):
    a: float = field()
    b: float = field()


def test_implementation_reuses_incidence_across_structural_and_backend_analysers() -> None:
    implementation = Implementation(SharedPorts, [connect])
    incidence = implementation.structural_analysis().incidence
    analyser = implementation.analyser_for(PYTHON_BACKEND)
    assert getattr(analyser, "full_incidence_graph") is incidence
    assert incidence.canonical_index is implementation.structural_analysis().incidence.canonical_index
