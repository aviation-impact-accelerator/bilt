import typing

from bilt import equation, transform
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@equation
def addition(a: float, b: float, c: float) -> bool:
    return a + b == c


@transform
def d(c: float) -> float:
    return c * 2


def connected_attributes(
    incidence: FullIncidenceGraph,
    constituent: BiltModelConstituent,
) -> set[str]:
    node = next(index for index in incidence.constituent_nodes if incidence.graph[index] is constituent)
    return {incidence.graph[neighbor] for neighbor in incidence.graph.neighbors(node)}


def test_equation_connects_to_all_parameters() -> None:
    incidence = FullIncidenceGraph([addition])

    assert connected_attributes(incidence, addition) == {"a", "b", "c"}


def test_transform_connects_to_inputs_and_output() -> None:
    incidence = FullIncidenceGraph([d])

    assert connected_attributes(incidence, d) == {"c", "d"}


def test_equation_and_transform_share_attribute() -> None:
    incidence = FullIncidenceGraph([addition, d])

    # Exactly one node represents the shared attribute.
    shared_nodes = [
        index
        for index in incidence.graph.node_indices()
        if isinstance(incidence.graph[index], str) and incidence.graph[index] == "c"
    ]
    assert len(shared_nodes) == 1

    neighbors = list(incidence.graph.neighbors(shared_nodes[0]))
    length_neighbours = 2

    assert len(neighbors) == length_neighbours
    assert any(incidence.graph[node] is addition for node in neighbors)
    assert any(incidence.graph[node] is d for node in neighbors)
