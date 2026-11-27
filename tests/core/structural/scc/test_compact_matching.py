"""Compact matching agrees with coordinate-level Dulmage-Mendelsohn analysis."""

import pytest

from bilt.core._domain_registry import DomainRegistry
from bilt.core._structural_analysis._causalization_block_solver._dm import dulmage_mendelsohn
from bilt.core._structural_analysis._compact_incidence import BiltCompactIncidenceError, build_compact_incidence
from bilt.core._structural_analysis._compact_matching import match_compact_incidence
from bilt.symbolic._atoms import ADD, MUL
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import domain
from bilt.symbolic._tree import Tree, equality_tree, int_tree, isel_tree, sum_tree, symbol_tree

SOURCE = dim("source")
TARGET = dim("target")
SCENARIO = dim("scenario")


def _registry(*, source: int = 3, target: int = 3, scenario: int = 2) -> DomainRegistry:
    result = DomainRegistry()
    result.bind_axis(coordinate_axis(SOURCE, range(source)))
    result.bind_axis(coordinate_axis(TARGET, range(target)))
    result.bind_axis(coordinate_axis(SCENARIO, range(scenario)))
    return result


def _dense() -> Tree:
    return equality_tree(
        sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("x"))), SOURCE),
        symbol_tree("rhs"),
    )


@pytest.mark.parametrize("size", (1, 3, 1000, 10000))
def test_pointwise_matching_has_one_independent_block_per_element(size: int) -> None:
    equation = equality_tree(
        Tree(ADD, (symbol_tree("x"), symbol_tree("offset"))),
        symbol_tree("rhs"),
    )
    incidence = build_compact_incidence(
        (equation,),
        {"x": domain(TARGET)},
        {"offset": domain(TARGET), "rhs": domain(TARGET)},
    )
    matched = match_compact_incidence(incidence, _registry(target=size), max_nodes=1, max_edges=1)
    assert matched.pattern == "pointwise"
    assert matched.well_determined
    assert matched.residual_coordinates == matched.unknown_coordinates == size
    assert matched.independent_blocks == size
    assert matched.partition is None


@pytest.mark.parametrize(
    ("sources", "targets", "status"),
    (
        (2, 2, "well-determined"),
        (3, 2, "under-determined"),
        (2, 3, "over-determined"),
        (1000, 1000, "well-determined"),
        (10000, 10000, "well-determined"),
    ),
)
def test_dense_operator_is_matched_without_expanding(sources: int, targets: int, status: str) -> None:
    incidence = build_compact_incidence(
        (_dense(),),
        {"x": domain(SOURCE)},
        {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)},
    )
    matched = match_compact_incidence(incidence, _registry(source=sources, target=targets), max_nodes=1, max_edges=1)
    assert matched.pattern == "complete"
    assert matched.status == status
    assert matched.residual_coordinates == targets
    assert matched.unknown_coordinates == sources
    assert matched.independent_blocks == (1 if sources == targets else None)


@pytest.mark.parametrize(("sources", "targets"), ((1, 1), (2, 2), (3, 2), (2, 3)))
def test_regular_dense_classification_agrees_with_explicit_matching(sources: int, targets: int) -> None:
    incidence = build_compact_incidence(
        (_dense(),),
        {"x": domain(SOURCE)},
        {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)},
    )
    bindings = _registry(source=sources, target=targets)
    matched = match_compact_incidence(incidence, bindings)
    graph, unknowns = incidence.materialize(bindings)
    exact = dulmage_mendelsohn(graph, unknowns)
    expected = "well-determined"
    if exact.underdetermined:
        expected = "under-determined"
    elif exact.overdetermined:
        expected = "over-determined"
    assert matched.status == expected


def test_scalar_broadcast_is_overdetermined_and_not_pointwise() -> None:
    equation = equality_tree(symbol_tree("alpha"), symbol_tree("rhs"))
    incidence = build_compact_incidence((equation,), {"alpha": domain()}, {"rhs": domain(TARGET)})
    matched = match_compact_incidence(incidence, _registry(target=1000), max_nodes=1)
    assert matched.pattern == "complete"
    assert matched.status == "over-determined"
    assert matched.independent_blocks is None


def test_selection_uses_bounded_exact_matching() -> None:
    equation = equality_tree(isel_tree(symbol_tree("x"), SOURCE, int_tree(0)), symbol_tree("rhs"))
    incidence = build_compact_incidence((equation,), {"x": domain(SOURCE)}, {"rhs": domain()})
    matched = match_compact_incidence(incidence, _registry(source=3))
    assert matched.pattern == "expanded"
    assert matched.status == "under-determined"
    assert matched.partition is not None


def test_unrelated_unknowns_do_not_get_matched_accidentally() -> None:
    equation = equality_tree(symbol_tree("x"), symbol_tree("rhs"))
    incidence = build_compact_incidence(
        (equation,), {"x": domain(TARGET), "y": domain(TARGET)}, {"rhs": domain(TARGET)}
    )
    matched = match_compact_incidence(incidence, _registry(target=3))
    assert matched.pattern == "expanded"
    assert matched.status == "under-determined"


def test_mixed_independent_and_coupled_unknowns_fall_back_to_exact_matching() -> None:
    equations = (
        equality_tree(symbol_tree("x"), symbol_tree("rhs")),
        equality_tree(
            sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("y"))), SOURCE),
            symbol_tree("rhs2"),
        ),
    )
    incidence = build_compact_incidence(
        equations,
        {"x": domain(TARGET), "y": domain(SOURCE)},
        {"rhs": domain(TARGET), "rhs2": domain(TARGET), "A": domain(TARGET, SOURCE)},
    )
    matched = match_compact_incidence(incidence, _registry(source=3, target=3))
    assert matched.pattern == "expanded"
    assert matched.well_determined


def test_mixed_axis_reduction_falls_back_instead_of_claiming_dense() -> None:
    expression = sum_tree(Tree(MUL, (symbol_tree("A"), symbol_tree("x"))), SOURCE)
    equation = equality_tree(expression, symbol_tree("rhs"))
    incidence = build_compact_incidence(
        (equation,),
        {"x": domain(SOURCE, SCENARIO)},
        {"A": domain(TARGET, SOURCE, SCENARIO), "rhs": domain(TARGET, SCENARIO)},
    )
    matched = match_compact_incidence(incidence, _registry(source=2, target=2, scenario=2))
    assert matched.pattern == "expanded"
    assert matched.well_determined


def test_bounded_fallback_rejects_large_unsupported_patterns() -> None:
    equation = equality_tree(isel_tree(symbol_tree("x"), SOURCE, int_tree(0)), symbol_tree("rhs"))
    incidence = build_compact_incidence((equation,), {"x": domain(SOURCE)}, {"rhs": domain()})
    with pytest.raises(BiltCompactIncidenceError, match="nodes"):
        match_compact_incidence(incidence, _registry(source=10000))


def test_zero_length_axes_use_exact_empty_graph_instead_of_regular_shortcut() -> None:
    incidence = build_compact_incidence(
        (_dense(),),
        {"x": domain(SOURCE)},
        {"A": domain(TARGET, SOURCE), "rhs": domain(TARGET)},
    )
    result = match_compact_incidence(incidence, _registry(source=0, target=0))
    assert result.pattern == "expanded"
    assert result.well_determined
    assert result.residual_coordinates == result.unknown_coordinates == 0
