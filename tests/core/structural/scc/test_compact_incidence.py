"""Compact tensor incidence is exactly expandable for small structural systems."""

# ruff: noqa: PLR2004 - Small coordinate counts are expected structural test results.

import typing

import pytest

if typing.TYPE_CHECKING:
    import rustworkx as rx

from bilt.core._domain_registry import DomainRegistry
from bilt.core._structural_analysis._compact_incidence import (
    BiltCompactIncidenceError,
    ResidualCoordinate,
    UnknownCoordinate,
    build_compact_incidence,
)
from bilt.symbolic._atoms import ADD, MUL
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import domain
from bilt.symbolic._tree import (
    Tree,
    equality_tree,
    int_tree,
    isel_tree,
    sum_tree,
    symbol_tree,
)

SOURCE = dim("source")
TARGET = dim("target")


def _registry(*, source: int = 3, target: int = 2) -> DomainRegistry:
    result = DomainRegistry()
    result.bind_axis(coordinate_axis(SOURCE, range(source)))
    result.bind_axis(coordinate_axis(TARGET, range(target)))
    return result


def _edges(graph: rx.PyGraph) -> set[tuple[ResidualCoordinate, UnknownCoordinate]]:
    return {
        (graph[first], graph[second]) if isinstance(graph[first], ResidualCoordinate) else (graph[second], graph[first])
        for first, second in graph.edge_list()
    }


def test_dense_source_target_operator_stays_compact() -> None:
    equation = equality_tree(
        sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("x"))), SOURCE),
        symbol_tree("rhs"),
    )
    incidence = build_compact_incidence(
        (equation,),
        {"x": domain(SOURCE)},
        {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)},
    )
    assert incidence.pattern_count == 1
    assert incidence.residuals[0].domain == domain(TARGET)
    graph, unknown_nodes = incidence.materialize(_registry(source=3, target=2))
    assert len(unknown_nodes) == 3
    assert len(_edges(graph)) == 6
    assert _edges(graph) == {
        (ResidualCoordinate(0, (target,)), UnknownCoordinate("x", (source,)))
        for target in range(2)
        for source in range(3)
    }


def test_scalar_broadcast_and_pointwise_tensor_dependencies() -> None:
    equation = equality_tree(
        Tree(ADD, (symbol_tree("x"), symbol_tree("alpha"))),
        symbol_tree("rhs"),
    )
    incidence = build_compact_incidence(
        (equation,),
        {"x": domain(TARGET), "alpha": domain()},
        {"rhs": domain(TARGET)},
    )
    assert incidence.pattern_count == 2
    assert incidence.residuals[0].dependencies((1,), _registry()) == {
        UnknownCoordinate("x", (1,)),
        UnknownCoordinate("alpha", ()),
    }
    graph, _ = incidence.materialize(_registry())
    assert len(_edges(graph)) == 4


def test_selection_fixes_one_coordinate_and_preserves_target_axis() -> None:
    equation = equality_tree(
        isel_tree(symbol_tree("x"), SOURCE, int_tree(-1)),
        symbol_tree("rhs"),
    )
    incidence = build_compact_incidence(
        (equation,),
        {"x": domain(TARGET, SOURCE)},
        {"rhs": domain(TARGET)},
    )
    assert incidence.residuals[0].dependencies((0,), _registry()) == {UnknownCoordinate("x", (0, 2))}
    assert incidence.residuals[0].dependencies((1,), _registry()) == {UnknownCoordinate("x", (1, 2))}


def test_two_scalar_rows_can_refer_to_different_tensor_coordinates() -> None:
    x = symbol_tree("x")
    equations = (
        equality_tree(isel_tree(x, SOURCE, int_tree(0)), symbol_tree("first")),
        equality_tree(isel_tree(x, SOURCE, int_tree(1)), symbol_tree("second")),
    )
    incidence = build_compact_incidence(
        equations,
        {"x": domain(SOURCE)},
        {"first": domain(), "second": domain()},
    )
    graph, _ = incidence.materialize(_registry(source=2))
    assert _edges(graph) == {
        (ResidualCoordinate(0, ()), UnknownCoordinate("x", (0,))),
        (ResidualCoordinate(1, ()), UnknownCoordinate("x", (1,))),
    }


def test_one_tensor_row_can_refer_to_two_scalar_unknowns() -> None:
    equation = equality_tree(
        Tree(ADD, (Tree(MUL, (symbol_tree("a"), symbol_tree("x"))), Tree(MUL, (symbol_tree("b"), symbol_tree("y"))))),
        symbol_tree("rhs"),
    )
    incidence = build_compact_incidence(
        (equation,),
        {"x": domain(), "y": domain()},
        {"a": domain(TARGET), "b": domain(TARGET), "rhs": domain(TARGET)},
    )
    assert len(incidence.residuals) == 1
    assert incidence.pattern_count == 2
    graph, _ = incidence.materialize(_registry())
    assert len(_edges(graph)) == 4


def test_duplicate_equations_keep_distinct_residual_identities() -> None:
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    incidence = build_compact_incidence(
        (equation, equation),
        {"x": domain()},
        {"rhs": domain()},
    )
    graph, _ = incidence.materialize(_registry())
    assert _edges(graph) == {
        (ResidualCoordinate(0, ()), UnknownCoordinate("x", ())),
        (ResidualCoordinate(1, ()), UnknownCoordinate("x", ())),
    }


def test_duplicate_paths_do_not_create_duplicate_graph_edges() -> None:
    equation = equality_tree(Tree(ADD, (symbol_tree("x"), symbol_tree("x"))), symbol_tree("rhs"))
    incidence = build_compact_incidence((equation,), {"x": domain()}, {"rhs": domain()})
    assert incidence.pattern_count == 2
    graph, _ = incidence.materialize(_registry())
    assert len(_edges(graph)) == 1


def test_zero_length_reduction_has_no_dependency_edges() -> None:
    equation = equality_tree(sum_tree(symbol_tree("x"), SOURCE), symbol_tree("rhs"))
    incidence = build_compact_incidence((equation,), {"x": domain(SOURCE)}, {"rhs": domain()})
    graph, unknown_nodes = incidence.materialize(_registry(source=0))
    assert graph.num_edges() == 0
    assert not unknown_nodes


def test_large_dense_system_cannot_be_materialized_unboundedly() -> None:
    equation = equality_tree(
        sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("x"))), SOURCE),
        symbol_tree("rhs"),
    )
    incidence = build_compact_incidence(
        (equation,), {"x": domain(SOURCE)}, {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)}
    )
    registry = _registry(source=10000, target=10000)
    assert incidence.pattern_count == 1
    with pytest.raises(BiltCompactIncidenceError, match="nodes"):
        incidence.materialize(registry)
    with pytest.raises(BiltCompactIncidenceError, match="edge limit"):
        incidence.materialize(_registry(source=3, target=3), max_nodes=100, max_edges=5)


def test_incomplete_and_ambiguous_domains_fail_explicitly() -> None:
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    with pytest.raises(ValueError, match="No domain supplied"):
        build_compact_incidence((equation,), {"x": domain()}, {})
    with pytest.raises(BiltCompactIncidenceError, match="both an unknown"):
        build_compact_incidence((equation,), {"x": domain()}, {"x": domain(), "rhs": domain()})


def test_non_equality_is_rejected() -> None:
    with pytest.raises(BiltCompactIncidenceError, match="binary equality"):
        build_compact_incidence((symbol_tree("x"),), {"x": domain()}, {})
