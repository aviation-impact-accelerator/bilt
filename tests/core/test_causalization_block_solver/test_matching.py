import pytest

from bilt import Implementation, Interface, field, system, transform
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._matching import build_matching_graph, find_matching
from bilt.core._sub_system import SubSystemConstituent


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
