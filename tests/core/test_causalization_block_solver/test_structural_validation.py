import pytest
import typing

from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_structure
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)

import rustworkx as rx

def test_well_constrained_problem_preserves_graph() -> None:
    graph = rx.PyGraph(multigraph=False)
    a = graph.add_node("a")
    x = graph.add_node("x")
    equation = graph.add_node("first")

    graph.add_edge(equation, a, None)
    graph.add_edge(equation, x, None)

    original_nodes = list(graph.nodes())
    original_edges = list(graph.edge_list())

    partition = validate_structure(
        graph,
        attribute_nodes=frozenset({a, x}),
        input_nodes=frozenset({a}),
    )

    assert partition.well_determined == frozenset({x, equation})
    assert not partition.underdetermined
    assert not partition.overdetermined
    assert list(graph.nodes()) == original_nodes
    assert list(graph.edge_list()) == original_edges


def test_underconstrained_problem_reports_attributes() -> None:
    graph = rx.PyGraph(multigraph=False)
    a = graph.add_node("a")
    x = graph.add_node("x")
    y = graph.add_node("y")
    equation = graph.add_node("first")

    for attribute in (a, x, y):
        graph.add_edge(equation, attribute, None)

    with pytest.raises(BiltUnderconstrainedError) as raised:
        validate_structure(
            graph,
            attribute_nodes=frozenset({a, x, y}),
            input_nodes=frozenset({a}),
        )

    assert "x" in str(raised.value)
    assert "y" in str(raised.value)


@pytest.mark.parametrize("include_underconstrained_region", [False, True])
def test_competing_constraints_are_rejected(
    include_underconstrained_region: bool,
) -> None:
    graph = rx.PyGraph(multigraph=False)
    a = graph.add_node("a")
    x = graph.add_node("x")
    first = graph.add_node("first")
    second = graph.add_node("second")

    for equation in (first, second):
        graph.add_edge(equation, a, None)
        graph.add_edge(equation, x, None)

    attributes = {a, x}
    if include_underconstrained_region:
        attributes.add(graph.add_node("y"))

    with pytest.raises(BiltOverconstrainedError) as raised:
        validate_structure(
            graph,
            attribute_nodes=frozenset(attributes),
            input_nodes=frozenset({a}),
        )

    assert "first" in str(raised.value)
    assert "second" in str(raised.value)
