"""Check bulk graph assembly against the original definitive incidence layout."""

import pytest
import rustworkx as rx

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._structural_analysis._causalization_block_solver import _structural_validation
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpandedConstraint,
    ExpansionSnapshot,
    StructuralExpansion,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_structure
from bilt.core._sub_system import SubSystemConstituent


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@equation
def relation(a: float, b: float) -> bool:
    return b == a + 1


@transform
def c(b: float) -> float:
    return b * 2


def _original_incidence(snapshot: ExpansionSnapshot) -> tuple[rx.PyGraph, dict[str, int]]:
    graph = rx.PyGraph(multigraph=False)
    attributes = {name: graph.add_node(name) for name in sorted(snapshot.attributes - snapshot.available)}
    for constraint in snapshot.constraints:
        node = graph.add_node(constraint.label)
        for attribute in sorted(constraint.attributes - snapshot.available):
            graph.add_edge(node, attributes[attribute], None)
    return graph, attributes


def _compare(snapshot: ExpansionSnapshot) -> None:
    bulk, actual = snapshot.to_incidence()
    original, expected = _original_incidence(snapshot)
    assert actual == expected
    assert list(bulk.nodes()) == list(original.nodes())
    assert list(bulk.edge_list()) == list(original.edge_list())
    assert list(bulk.node_indices()) == list(original.node_indices())


@pytest.mark.parametrize("supplied", (frozenset(), frozenset({"a"}), frozenset({"a", "c"})))
def test_bulk_incidence_preserves_exact_order_and_edges(supplied: frozenset[str]) -> None:
    rows = (
        ExpandedConstraint((), relation, frozenset({"a", "b"})),
        ExpandedConstraint((), c, frozenset({"b", "c"})),
    )
    snapshot = ExpansionSnapshot(
        constraints=rows,
        attributes=frozenset({"a", "b", "c"}),
        available=supplied,
        unresolved=(),
        expanded_paths=frozenset(),
        query_count=0,
    )
    _compare(snapshot)


def test_bulk_incidence_preserves_distinct_shared_child_instances() -> None:
    shared = Implementation(Ports, [relation, c])
    left = SubSystemConstituent("left", shared)
    right = SubSystemConstituent("right", shared)

    @equation
    def combine(answer: float, left: Ports, right: Ports) -> bool:
        return answer == left.c + right.c

    expansion = StructuralExpansion(FullIncidenceGraph([left, right, combine]), {"left.a", "right.a"}, "answer")
    expansion.collect()
    snapshot = expansion.snapshot()
    assert snapshot.complete
    assert snapshot.expanded_paths == frozenset({("left",), ("right",)})
    _compare(snapshot)


def test_validation_does_not_copy_when_no_inputs_are_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = rx.PyGraph(multigraph=False)
    attribute = graph.add_node("a")
    operation = graph.add_node("relation")
    graph.add_edge(operation, attribute, None)

    calls: list[rx.PyGraph] = []
    original = _structural_validation.dulmage_mendelsohn

    def inspect(received: rx.PyGraph, nodes: frozenset[int]) -> _structural_validation.DMPartition:
        calls.append(received)
        return original(received, nodes)

    monkeypatch.setattr(_structural_validation, "dulmage_mendelsohn", inspect)
    validate_structure(graph, frozenset({attribute}), frozenset())
    assert calls == [graph]


def test_validation_copies_when_inputs_are_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = rx.PyGraph(multigraph=False)
    supplied = graph.add_node("supplied")
    variable = graph.add_node("variable")
    operation = graph.add_node("relation")
    graph.add_edge(operation, supplied, None)
    graph.add_edge(operation, variable, None)

    copies: list[rx.PyGraph] = []
    original = _structural_validation.dulmage_mendelsohn

    def inspect(received: rx.PyGraph, nodes: frozenset[int]) -> _structural_validation.DMPartition:
        copies.append(received)
        return original(received, nodes)

    monkeypatch.setattr(_structural_validation, "dulmage_mendelsohn", inspect)
    validate_structure(graph, frozenset({supplied, variable}), frozenset({supplied}))
    assert len(copies) == 1
    assert copies[0] is not graph
    assert graph.num_nodes() == len({supplied, variable, operation})
    assert graph.num_edges() == len({supplied, variable})
