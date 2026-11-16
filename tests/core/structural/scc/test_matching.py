import typing

import pytest
import rustworkx as rx

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpandedConstraint,
    ExpansionSnapshot,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._matching import (
    _forward_producers,
    build_matching_graph,
    find_matching,
)
from bilt.core._sub_system import SubSystemConstituent

if typing.TYPE_CHECKING:
    from bilt.core._equation import BasicEquation


@pytest.fixture
def child() -> SubSystemConstituent:
    @system
    class Child(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def b(a: float) -> float:
        return a * 2

    @transform
    def c(a: float) -> float:
        return a + 1

    return SubSystemConstituent(
        name="child",
        implementation=Implementation(Child, [b, c]),
    )


def test_matching_assigns_multiple_subsystem_outputs(
    child: SubSystemConstituent,
) -> None:
    incidence = FullIncidenceGraph([child])

    graph = build_matching_graph(incidence, {"child.a"})
    assignments = find_matching(graph)

    assert set(assignments) == {"child.b", "child.c"}
    assert assignments["child.b"] is child
    assert assignments["child.c"] is child


def test_matching_excludes_supplied_subsystem_ports(
    child: SubSystemConstituent,
) -> None:
    incidence = FullIncidenceGraph([child])

    graph = build_matching_graph(incidence, {"child.a", "child.b"})
    assignments = find_matching(graph)

    assert set(assignments) == {"child.c"}
    assert assignments["child.c"] is child


def test_subsystem_matching_preserves_full_incidence(
    child: SubSystemConstituent,
) -> None:
    incidence = FullIncidenceGraph([child])
    original_nodes = list(incidence.graph.nodes())
    original_edges = set(incidence.graph.edge_list())

    build_matching_graph(incidence, {"child.a"})

    assert list(incidence.graph.nodes()) == original_nodes
    assert set(incidence.graph.edge_list()) == original_edges


def test_matching_prefers_ready_parent_producer_without_removing_child_alternative() -> None:
    @system
    class Child(metaclass=Interface):
        a: float = field()
        b: float = field()

    child = SubSystemConstituent("child", Implementation(Child, []))

    @equation
    def supply(source: float, child: Child) -> bool:
        return child.a == source

    incidence = FullIncidenceGraph([child, supply])
    graph = build_matching_graph(incidence, {"source"})
    assert find_matching(graph)["child.a"] is supply
    graph.remove_node(next(node for node in graph.node_indices() if graph[node] is supply))
    assert find_matching(graph)["child.a"] is child


def test_matching_priority_never_reduces_cardinality() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        return x == y + 1

    @transform
    def x(a: float) -> float:
        return a + 1

    graph = rx.PyGraph(multigraph=False)
    x_node = graph.add_node("x")
    y_node = graph.add_node("y")
    relation_node = graph.add_node(relation)
    transform_node = graph.add_node(x)
    graph.add_edge(relation_node, x_node, 1)
    graph.add_edge(relation_node, y_node, 0)
    graph.add_edge(transform_node, x_node, 0)
    assignments = find_matching(graph)
    assert assignments == {"x": x, "y": relation}


def test_forward_producers_recognises_small_local_linear_block() -> None:
    @equation
    def total(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def difference(b: float, x: float, y: float) -> bool:
        return x - y == b

    constraints = (
        ExpandedConstraint(("inner",), total, frozenset({"inner.a", "inner.x", "inner.y"})),
        ExpandedConstraint(("inner",), difference, frozenset({"inner.b", "inner.x", "inner.y"})),
    )
    snapshot = ExpansionSnapshot(
        constraints=constraints,
        attributes=frozenset({"inner.a", "inner.b", "inner.x", "inner.y"}),
        available=frozenset({"inner.a", "inner.b"}),
        unresolved=(),
        expanded_paths=frozenset({("inner",)}),
        query_count=0,
    )

    producers = _forward_producers(snapshot)
    assert set(producers) == {"inner.x", "inner.y"}
    assert set(producers.values()) == set(constraints)


def test_forward_producers_does_not_prefer_nonlinear_coupling() -> None:
    @equation
    def total(a: float, x: float, y: float) -> bool:
        return x * y == a

    @equation
    def difference(b: float, x: float, y: float) -> bool:
        return x - y == b

    constraints = (
        ExpandedConstraint((), total, frozenset({"a", "x", "y"})),
        ExpandedConstraint((), difference, frozenset({"b", "x", "y"})),
    )
    snapshot = ExpansionSnapshot(
        constraints=constraints,
        attributes=frozenset({"a", "b", "x", "y"}),
        available=frozenset({"a", "b"}),
        unresolved=(),
        expanded_paths=frozenset(),
        query_count=0,
    )

    assert _forward_producers(snapshot) == {}


@pytest.mark.parametrize("size", (3, 4))
def test_forward_producers_recognises_larger_sparse_linear_blocks(size: int) -> None:
    four_variables = 4

    @equation
    def first(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def second(b: float, y: float, z: float) -> bool:
        return y + z == b

    @equation
    def third(c: float, x: float, z: float) -> bool:
        return x + z == c

    @equation
    def third_four(c: float, z: float, w: float) -> bool:
        return z + w == c

    @equation
    def fourth(d: float, w: float, x: float) -> bool:
        return w + 2 * x == d

    pairs: list[tuple[BasicEquation, set[str]]] = [
        (first, {"a", "x", "y"}),
        (second, {"b", "y", "z"}),
        (third, {"c", "x", "z"}),
    ]
    if size == four_variables:
        pairs[2] = (third_four, {"c", "z", "w"})
        pairs.append((fourth, {"d", "w", "x"}))
    constraints = tuple(ExpandedConstraint((), source, frozenset(names)) for source, names in pairs)
    known = frozenset({"a", "b", "c", "d"} if size == four_variables else {"a", "b", "c"})
    unknown = {"x", "y", "z", "w"} if size == four_variables else {"x", "y", "z"}
    snapshot = ExpansionSnapshot(
        constraints=constraints,
        attributes=frozenset(known | unknown),
        available=known,
        unresolved=(),
        expanded_paths=frozenset(),
        query_count=0,
    )

    producers = _forward_producers(snapshot)
    assert set(producers) == unknown
    assert set(producers.values()) == set(constraints)


def test_forward_producers_recognises_mixed_parent_child_linear_block() -> None:
    @system
    class Child(metaclass=Interface):
        a: float = field()
        x: float = field()
        y: float = field()

    @equation
    def child_relation(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def parent_link(child: Child, b: float) -> bool:
        return child.x - child.y == b

    constraints = (
        ExpandedConstraint(("child",), child_relation, frozenset({"child.a", "child.x", "child.y"})),
        ExpandedConstraint((), parent_link, frozenset({"child.x", "child.y", "b"})),
    )
    snapshot = ExpansionSnapshot(
        constraints=constraints,
        attributes=frozenset({"child.a", "child.x", "child.y", "b"}),
        available=frozenset({"child.a", "b"}),
        unresolved=(),
        expanded_paths=frozenset({("child",)}),
        query_count=0,
    )
    producers = _forward_producers(snapshot)
    assert set(producers) == {"child.x", "child.y"}


def test_forward_producers_does_not_hint_incomplete_coupled_group() -> None:
    @equation
    def first(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def second(b: float, y: float, z: float) -> bool:
        return y + z == b

    constraints = (
        ExpandedConstraint((), first, frozenset({"a", "x", "y"})),
        ExpandedConstraint((), second, frozenset({"b", "y", "z"})),
    )
    snapshot = ExpansionSnapshot(
        constraints=constraints,
        attributes=frozenset({"a", "b", "x", "y", "z"}),
        available=frozenset({"a", "b"}),
        unresolved=(),
        expanded_paths=frozenset(),
        query_count=0,
    )
    assert _forward_producers(snapshot) == {}
