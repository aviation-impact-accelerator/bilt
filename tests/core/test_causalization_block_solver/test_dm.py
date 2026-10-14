import pytest
import rustworkx as rx

from bilt.core._structural_analysis._causalization_block_solver._dm import (
    dulmage_mendelsohn,
)


def test_square_system_is_well_determined() -> None:
    graph = rx.PyGraph(multigraph=False)
    x, y = graph.add_nodes_from(("x", "y"))
    first, second = graph.add_nodes_from((object(), object()))
    graph.add_edges_from_no_data(
        (
            (first, x),
            (first, y),
            (second, x),
            (second, y),
        ),
    )

    result = dulmage_mendelsohn(graph, {x, y})

    assert result.underdetermined == frozenset()
    assert result.well_determined == frozenset({x, y, first, second})
    assert result.overdetermined == frozenset()


def test_underdetermined_region_includes_matched_nodes() -> None:
    graph = rx.PyGraph(multigraph=False)
    x, y = graph.add_nodes_from(("x", "y"))
    operation = graph.add_node(object())
    graph.add_edges_from_no_data(
        ((operation, x), (operation, y)),
    )

    result = dulmage_mendelsohn(graph, {x, y})

    assert result.underdetermined == frozenset({x, y, operation})
    assert result.well_determined == frozenset()
    assert result.overdetermined == frozenset()


def test_overdetermined_region_includes_matched_nodes() -> None:
    graph = rx.PyGraph(multigraph=False)
    x = graph.add_node("x")
    first, second = graph.add_nodes_from((object(), object()))
    graph.add_edges_from_no_data(
        ((first, x), (second, x)),
    )

    result = dulmage_mendelsohn(graph, {x})

    assert result.underdetermined == frozenset()
    assert result.well_determined == frozenset()
    assert result.overdetermined == frozenset({x, first, second})


def test_equal_counts_can_hide_both_deficient_regions() -> None:
    graph = rx.PyGraph(multigraph=False)
    x, y, z = graph.add_nodes_from(("x", "y", "z"))
    first, second, third = graph.add_nodes_from(
        (object(), object(), object()),
    )
    graph.add_edges_from_no_data(
        (
            (first, x),
            (first, y),
            (second, z),
            (third, z),
        ),
    )

    result = dulmage_mendelsohn(graph, {x, y, z})

    assert result.underdetermined == frozenset({x, y, first})
    assert result.well_determined == frozenset()
    assert result.overdetermined == frozenset({z, second, third})


def test_connected_graph_can_contain_all_three_regions() -> None:
    graph = rx.PyGraph(multigraph=False)
    x, y, z, w = graph.add_nodes_from(("x", "y", "z", "w"))
    first, second, third, fourth = graph.add_nodes_from(
        (object(), object(), object(), object()),
    )
    graph.add_edges_from_no_data(
        (
            (first, x),
            (first, y),
            (first, z),
            (second, z),
            (second, w),
            (third, w),
            (fourth, w),
        ),
    )

    result = dulmage_mendelsohn(graph, {x, y, z, w})

    assert result.underdetermined == frozenset({x, y, first})
    assert result.well_determined == frozenset({z, second})
    assert result.overdetermined == frozenset({w, third, fourth})


def test_isolated_nodes_are_classified() -> None:
    graph = rx.PyGraph(multigraph=False)
    x = graph.add_node("x")
    operation = graph.add_node(object())

    result = dulmage_mendelsohn(graph, {x})

    assert result.underdetermined == frozenset({x})
    assert result.well_determined == frozenset()
    assert result.overdetermined == frozenset({operation})


def test_empty_graph_has_empty_regions() -> None:
    result = dulmage_mendelsohn(
        rx.PyGraph(multigraph=False),
        set(),
    )

    assert result.underdetermined == frozenset()
    assert result.well_determined == frozenset()
    assert result.overdetermined == frozenset()


def test_partition_handles_removed_node_indices_without_mutation() -> None:
    graph = rx.PyGraph(multigraph=False)
    removed = graph.add_node(object())
    x = graph.add_node("x")
    operation = graph.add_node(object())
    graph.add_edge(operation, x, None)
    graph.remove_node(removed)
    original_nodes = list(graph.node_indices())
    original_edges = list(graph.edge_list())

    result = dulmage_mendelsohn(graph, {x})

    assert result.well_determined == frozenset({x, operation})
    assert list(graph.node_indices()) == original_nodes
    assert list(graph.edge_list()) == original_edges


def test_partition_rejects_edges_within_one_side() -> None:
    graph = rx.PyGraph(multigraph=False)
    x, y = graph.add_nodes_from(("x", "y"))
    graph.add_edge(x, y, None)

    with pytest.raises(ValueError, match="connect an attribute"):
        dulmage_mendelsohn(graph, {x, y})


def test_partition_rejects_unknown_attribute_nodes() -> None:
    graph = rx.PyGraph(multigraph=False)
    removed = graph.add_node("removed")
    graph.remove_node(removed)

    with pytest.raises(ValueError, match="belong to the graph"):
        dulmage_mendelsohn(graph, {removed})