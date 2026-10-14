from bilt import equation, transform
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._matching import (
    build_matching_graph,
)


@equation
def addition(a: float, b: float, c: float) -> bool:
    return a + b == c


@transform
def d(c: float) -> float:
    return c * 2


def test_matching_excludes_supplied_attributes() -> None:
    incidence = FullIncidenceGraph([addition, d])

    graph = build_matching_graph(incidence, {"a", "b"})

    attributes = {graph[node] for node in graph.node_indices() if isinstance(graph[node], str)}
    assert attributes == {"c", "d"}


def test_matching_retains_equation_candidates() -> None:
    incidence = FullIncidenceGraph([addition])

    graph = build_matching_graph(incidence, {"a"})

    equation_node = next(node for node in graph.node_indices() if graph[node] is addition)
    assert {graph[node] for node in graph.neighbors(equation_node)} == {"b", "c"}


def test_matching_limits_transform_to_output() -> None:
    incidence = FullIncidenceGraph([d])

    graph = build_matching_graph(incidence, set())

    transform_node = next(node for node in graph.node_indices() if graph[node] is d)
    assert {graph[node] for node in graph.neighbors(transform_node)} == {"d"}


def test_transform_with_supplied_output_has_no_assignment() -> None:
    incidence = FullIncidenceGraph([d])

    graph = build_matching_graph(incidence, {"d"})

    assert graph.num_edges() == 0


def test_matching_preserves_full_incidence() -> None:
    incidence = FullIncidenceGraph([addition, d])
    original_nodes = list(incidence.graph.nodes())
    original_edges = set(incidence.graph.edge_list())

    build_matching_graph(incidence, {"a", "b"})

    assert list(incidence.graph.nodes()) == original_nodes
    assert set(incidence.graph.edge_list()) == original_edges
