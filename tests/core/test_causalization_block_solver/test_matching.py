from bilt import equation, transform
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._matching import build_matching_graph, find_matching


@equation
def addition(a: float, b: float, c: float) -> bool:
    return a + b == c


@transform
def d(c: float) -> float:
    return c * 2


@transform
def c(a: float) -> float:
    return a * 2


@equation
def multiplication(a: float, b: float, c: float) -> bool:
    return a * b == c


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


def test_matching_assigns_equation_and_transform() -> None:
    incidence = FullIncidenceGraph([addition, d])
    graph = build_matching_graph(incidence, {"a", "b"})

    assignments = find_matching(graph)

    assert set(assignments) == {"c", "d"}
    assert assignments["c"] is addition
    assert assignments["d"] is d


def test_matching_finds_maximum_number_of_assignments() -> None:
    # The equation could produce b or c, but the transform only produces c.
    # Both can be used if the equation is assigned to b.
    incidence = FullIncidenceGraph([addition, c])
    graph = build_matching_graph(incidence, {"a"})

    assignments = find_matching(graph)

    assert set(assignments) == {"b", "c"}
    assert assignments["b"] is addition
    assert assignments["c"] is c


def test_matching_can_return_partial_assignment() -> None:
    # One equation cannot independently determine both b and c.
    incidence = FullIncidenceGraph([addition])
    graph = build_matching_graph(incidence, {"a"})

    assignments = find_matching(graph)

    assert len(assignments) == 1
    assert set(assignments) <= {"b", "c"}
    assert next(iter(assignments.values())) is addition


def test_matching_allows_surplus_constituents() -> None:
    incidence = FullIncidenceGraph([addition, multiplication, d])
    graph = build_matching_graph(incidence, {"a", "b"})

    assignments = find_matching(graph)

    assert set(assignments) == {"c", "d"}
    assert assignments["c"] is addition or assignments["c"] is multiplication
    assert assignments["d"] is d


def test_matching_with_no_unknowns_returns_empty_assignment() -> None:
    incidence = FullIncidenceGraph([addition, d])
    graph = build_matching_graph(incidence, {"a", "b", "c", "d"})

    assert find_matching(graph) == {}
