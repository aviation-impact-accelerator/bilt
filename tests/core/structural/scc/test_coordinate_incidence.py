"""Tests for symbolic coordinate incidence before integration into planning."""

# ruff: noqa: PLR2004 - Coordinate counts and small panel sizes are test expectations.

import pytest

from bilt.core._domain_registry import DomainRegistry
from bilt.core._structural_analysis._causalization_block_solver._coordinate_incidence import (
    ResidualCoordinate,
    UnknownCoordinate,
    build_coordinate_incidence,
)
from bilt.core._structural_analysis._causalization_block_solver._dm import dulmage_mendelsohn
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.symbolic._atoms import ADD, MUL
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import domain
from bilt.symbolic._tree import (
    Tree,
    coordinate_tree,
    equality_tree,
    int_tree,
    isel_tree,
    mean_tree,
    sel_tree,
    sum_tree,
    symbol_tree,
)

SOURCE = dim("source")
TARGET = dim("target")
WING_SOURCE = dim("wing_source")
TAIL_SOURCE = dim("tail_source")
WING_TARGET = dim("wing_target")
TAIL_TARGET = dim("tail_target")


def _registry(**extents: int) -> DomainRegistry:
    result = DomainRegistry()
    for name, size in extents.items():
        dimension = {
            "source": SOURCE,
            "target": TARGET,
            "wing_source": WING_SOURCE,
            "tail_source": TAIL_SOURCE,
            "wing_target": WING_TARGET,
            "tail_target": TAIL_TARGET,
        }[name]
        result.bind_axis(coordinate_axis(dimension, range(size)))
    return result


def _mul(left: Tree, right: Tree) -> Tree:
    return Tree(MUL, (left, right))


def _add(*values: Tree) -> Tree:
    return Tree(ADD, values)


def test_source_to_target_contraction_has_dense_structural_incidence() -> None:
    """Two target rows each depend on both source coordinates."""
    gamma = symbol_tree("gamma")
    equation = equality_tree(sum_tree(_mul(symbol_tree("influence"), gamma), SOURCE), symbol_tree("rhs"))
    result = build_coordinate_incidence(
        (equation,),
        {"gamma": domain(SOURCE)},
        {"influence": domain(TARGET, SOURCE), "rhs": domain(TARGET)},
        _registry(source=2, target=2),
    )
    assert result.unknowns == (UnknownCoordinate("gamma", (0,)), UnknownCoordinate("gamma", (1,)))
    assert result.residuals == (ResidualCoordinate(0, (0,)), ResidualCoordinate(0, (1,)))
    assert len(result.edges) == 4
    graph, unknown_nodes = result.to_graph()
    partition = dulmage_mendelsohn(graph, unknown_nodes)
    assert len(partition.well_determined) == 4
    assert not partition.underdetermined
    assert not partition.overdetermined


def test_two_scalar_equations_determine_one_tensor() -> None:
    """Primitive equation count may exceed unknown port count."""
    x = symbol_tree("x")
    equations = (
        equality_tree(isel_tree(x, SOURCE, int_tree(0)), symbol_tree("first")),
        equality_tree(isel_tree(x, SOURCE, int_tree(1)), symbol_tree("second")),
    )
    result = build_coordinate_incidence(
        equations,
        {"x": domain(SOURCE)},
        {"first": domain(), "second": domain()},
        _registry(source=2),
    )
    assert result.edges == frozenset(
        {
            (ResidualCoordinate(0, ()), UnknownCoordinate("x", (0,))),
            (ResidualCoordinate(1, ()), UnknownCoordinate("x", (1,))),
        }
    )
    graph, unknown_nodes = result.to_graph()
    assert not dulmage_mendelsohn(graph, unknown_nodes).underdetermined


def test_one_tensor_equation_determines_two_scalars() -> None:
    """One two-coordinate residual may independently constrain two scalars."""
    equation = equality_tree(
        _add(_mul(symbol_tree("a"), symbol_tree("x")), _mul(symbol_tree("b"), symbol_tree("y"))),
        symbol_tree("rhs"),
    )
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(), "y": domain()},
        {"a": domain(TARGET), "b": domain(TARGET), "rhs": domain(TARGET)},
        _registry(target=2),
    )
    assert len(result.residuals) == len(result.unknowns) == 2
    assert len(result.edges) == 4


def test_mixed_wing_tail_trim_is_six_by_six() -> None:
    """Four primitive rows form a six-DOF wing-tail constraint structure."""
    wing = symbol_tree("wing")
    tail = symbol_tree("tail")
    alpha = symbol_tree("alpha")
    elevator = symbol_tree("elevator")
    wing_boundary = equality_tree(
        _add(
            sum_tree(_mul(symbol_tree("ww"), wing), WING_SOURCE),
            sum_tree(_mul(symbol_tree("wt"), tail), TAIL_SOURCE),
            alpha,
        ),
        symbol_tree("wing_rhs"),
    )
    tail_boundary = equality_tree(
        _add(
            sum_tree(_mul(symbol_tree("tw"), wing), WING_SOURCE),
            sum_tree(_mul(symbol_tree("tt"), tail), TAIL_SOURCE),
            alpha,
            elevator,
        ),
        symbol_tree("tail_rhs"),
    )
    total = equality_tree(
        _add(
            sum_tree(_mul(symbol_tree("wing_width"), wing), WING_SOURCE),
            sum_tree(_mul(symbol_tree("tail_width"), tail), TAIL_SOURCE),
        ),
        symbol_tree("total"),
    )
    moment = equality_tree(
        _add(
            sum_tree(_mul(symbol_tree("wing_moment"), wing), WING_SOURCE),
            sum_tree(_mul(symbol_tree("tail_moment"), tail), TAIL_SOURCE),
        ),
        symbol_tree("moment"),
    )
    result = build_coordinate_incidence(
        (wing_boundary, tail_boundary, total, moment),
        {
            "wing": domain(WING_SOURCE),
            "tail": domain(TAIL_SOURCE),
            "alpha": domain(),
            "elevator": domain(),
        },
        {
            "ww": domain(WING_TARGET, WING_SOURCE),
            "wt": domain(WING_TARGET, TAIL_SOURCE),
            "tw": domain(TAIL_TARGET, WING_SOURCE),
            "tt": domain(TAIL_TARGET, TAIL_SOURCE),
            "wing_rhs": domain(WING_TARGET),
            "tail_rhs": domain(TAIL_TARGET),
            "wing_width": domain(WING_SOURCE),
            "tail_width": domain(TAIL_SOURCE),
            "wing_moment": domain(WING_SOURCE),
            "tail_moment": domain(TAIL_SOURCE),
            "total": domain(),
            "moment": domain(),
        },
        _registry(wing_source=2, tail_source=2, wing_target=2, tail_target=2),
    )
    assert len(result.unknowns) == len(result.residuals) == 6
    assert tuple(len([r for r in result.residuals if r.equation == i]) for i in range(4)) == (2, 2, 1, 1)
    graph, unknown_nodes = result.to_graph()
    partition = dulmage_mendelsohn(graph, unknown_nodes)
    assert not partition.underdetermined
    assert not partition.overdetermined


def test_pointwise_does_not_connect_unrelated_coordinates() -> None:
    """An elementwise equality is diagonal even though tensors have many entries."""
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(SOURCE)},
        {"rhs": domain(SOURCE)},
        _registry(source=3),
    )
    assert result.edges == frozenset((ResidualCoordinate(0, (i,)), UnknownCoordinate("x", (i,))) for i in range(3))


def test_broadcast_scalar_connects_to_each_residual() -> None:
    equation = equality_tree(_add(symbol_tree("x"), symbol_tree("alpha")), symbol_tree("rhs"))
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(SOURCE), "alpha": domain()},
        {"rhs": domain(SOURCE)},
        _registry(source=2),
    )
    assert (ResidualCoordinate(0, (0,)), UnknownCoordinate("alpha", ())) in result.edges
    assert (ResidualCoordinate(0, (1,)), UnknownCoordinate("alpha", ())) in result.edges
    graph, unknown_nodes = result.to_graph()
    partition = dulmage_mendelsohn(graph, unknown_nodes)
    assert partition.underdetermined


def test_reduction_connects_every_source_coordinate() -> None:
    equation = equality_tree(sum_tree(symbol_tree("x"), SOURCE), symbol_tree("rhs"))
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(SOURCE)},
        {"rhs": domain()},
        _registry(source=3),
    )
    assert len(result.edges) == 3
    graph, unknown_nodes = result.to_graph()
    assert dulmage_mendelsohn(graph, unknown_nodes).underdetermined


def test_duplicate_expressions_retained_as_distinct_residuals() -> None:
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    result = build_coordinate_incidence(
        (equation, equation),
        {"x": domain()},
        {"rhs": domain()},
        _registry(),
    )
    assert len(result.residuals) == 2
    graph, unknown_nodes = result.to_graph()
    assert dulmage_mendelsohn(graph, unknown_nodes).overdetermined


def test_coordinate_limit_fails_explicitly() -> None:
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    with pytest.raises(BiltLinearBlockError, match="coordinate limit"):
        build_coordinate_incidence(
            (equation,),
            {"x": domain(SOURCE)},
            {"rhs": domain(SOURCE)},
            _registry(source=10),
            max_coordinates=4,
        )


def test_edge_limit_fails_explicitly() -> None:
    equation = equality_tree(
        sum_tree(_mul(symbol_tree("influence"), symbol_tree("x")), SOURCE),
        symbol_tree("rhs"),
    )
    with pytest.raises(BiltLinearBlockError, match="edge limit"):
        build_coordinate_incidence(
            (equation,),
            {"x": domain(SOURCE)},
            {"influence": domain(TARGET, SOURCE), "rhs": domain(TARGET)},
            _registry(source=3, target=3),
            max_edges=4,
        )


def test_mean_uses_all_source_coordinates() -> None:
    equation = equality_tree(mean_tree(symbol_tree("x"), SOURCE), symbol_tree("rhs"))
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(SOURCE)},
        {"rhs": domain()},
        _registry(source=3),
    )
    assert len(result.edges) == 3


def test_label_selection_uses_registered_coordinates() -> None:
    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(SOURCE, ("inboard", "outboard")))
    equation = equality_tree(
        sel_tree(symbol_tree("x"), SOURCE, coordinate_tree("outboard")),
        symbol_tree("rhs"),
    )
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(SOURCE)},
        {"rhs": domain()},
        registry,
    )
    assert result.edges == frozenset({(ResidualCoordinate(0, ()), UnknownCoordinate("x", (1,)))})


def test_negative_index_selection_uses_last_coordinate() -> None:
    equation = equality_tree(
        isel_tree(symbol_tree("x"), SOURCE, int_tree(-1)),
        symbol_tree("rhs"),
    )
    result = build_coordinate_incidence(
        (equation,),
        {"x": domain(SOURCE)},
        {"rhs": domain()},
        _registry(source=3),
    )
    assert result.edges == frozenset({(ResidualCoordinate(0, ()), UnknownCoordinate("x", (2,)))})


def test_unknown_domain_must_be_resolved_before_refinement() -> None:
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    with pytest.raises(BiltLinearBlockError, match="No domain supplied"):
        build_coordinate_incidence(
            (equation,),
            {"y": domain()},
            {},
            _registry(),
        )
