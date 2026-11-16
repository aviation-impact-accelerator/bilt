"""Pure operator semantics must match existing named-axis lowering."""

import pytest

from bilt.core._domain_registry import DomainRegistry
from bilt.core._tensors._expression_lowering import lower_expression
from bilt.symbolic._atoms import ADD, MUL, NEG, POW, axis
from bilt.symbolic._dimensions import coordinate_axis, dim
from bilt.symbolic._domains import domain
from bilt.symbolic._expressions import Expr
from bilt.symbolic._operator_semantics import (
    BiltOperatorSemanticsError,
    SymbolCoordinate,
    describe_expression,
)
from bilt.symbolic._tree import (
    Tree,
    coordinate_tree,
    int_tree,
    isel_tree,
    mean_tree,
    sel_tree,
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


def test_contraction_retains_lazy_source_to_target_dependencies() -> None:
    product = Tree(MUL, (symbol_tree("matrix"), symbol_tree("x")))
    expression = sum_tree(product, SOURCE)
    semantics = describe_expression(
        expression,
        {"matrix": domain(TARGET, SOURCE), "x": domain(SOURCE)},
    )
    assert semantics.domain == domain(TARGET)
    assert list(semantics.dependencies((1,), _registry())) == [
        SymbolCoordinate("matrix", (1, 0)),
        SymbolCoordinate("x", (0,)),
        SymbolCoordinate("matrix", (1, 1)),
        SymbolCoordinate("x", (1,)),
        SymbolCoordinate("matrix", (1, 2)),
        SymbolCoordinate("x", (2,)),
    ]


def test_broadcast_scalar_affects_every_target_coordinate() -> None:
    expression = Tree(ADD, (symbol_tree("x"), symbol_tree("alpha")))
    semantics = describe_expression(
        expression,
        {"x": domain(TARGET), "alpha": domain()},
    )
    assert semantics.domain == domain(TARGET)
    assert set(semantics.dependencies((0,), _registry())) == {
        SymbolCoordinate("x", (0,)),
        SymbolCoordinate("alpha", ()),
    }
    assert set(semantics.dependencies((1,), _registry())) == {
        SymbolCoordinate("x", (1,)),
        SymbolCoordinate("alpha", ()),
    }


@pytest.mark.parametrize(("axis", "expected"), ((0, 0), (2, 2), (-1, 2)))
def test_literal_index_selection(axis: int, expected: int) -> None:
    semantics = describe_expression(
        isel_tree(symbol_tree("x"), SOURCE, int_tree(axis)),
        {"x": domain(TARGET, SOURCE)},
    )
    assert semantics.domain == domain(TARGET)
    assert list(semantics.dependencies((1,), _registry())) == [SymbolCoordinate("x", (1, expected))]


def test_label_selection_uses_bound_axis() -> None:
    registry = DomainRegistry()
    registry.bind_axis(coordinate_axis(SOURCE, ("root", "tip")))
    registry.bind_axis(coordinate_axis(TARGET, (0, 1)))
    semantics = describe_expression(
        sel_tree(symbol_tree("x"), SOURCE, coordinate_tree("tip")),
        {"x": domain(TARGET, SOURCE)},
    )
    assert list(semantics.dependencies((0,), registry)) == [SymbolCoordinate("x", (0, 1))]


def test_missing_or_out_of_range_selection_is_rejected() -> None:
    with pytest.raises(BiltOperatorSemanticsError, match="absent dimension"):
        describe_expression(
            isel_tree(symbol_tree("x"), SOURCE, int_tree(0)),
            {"x": domain(TARGET)},
        )
    semantics = describe_expression(
        isel_tree(symbol_tree("x"), SOURCE, int_tree(3)),
        {"x": domain(SOURCE)},
    )
    with pytest.raises(BiltOperatorSemanticsError, match="outside dimension"):
        list(semantics.dependencies((), _registry()))


def test_missing_named_symbol_is_not_silently_scalar() -> None:
    with pytest.raises(BiltOperatorSemanticsError, match="No domain supplied"):
        describe_expression(Tree(NEG, (symbol_tree("missing"),)), {})


def test_zero_length_reduction_has_no_structural_dependencies() -> None:
    semantics = describe_expression(
        sum_tree(symbol_tree("x"), SOURCE),
        {"x": domain(SOURCE)},
    )
    assert semantics.domain == domain()
    assert list(semantics.dependencies((), _registry(source=0))) == []


@pytest.mark.parametrize(
    "expression",
    (
        Tree(ADD, (symbol_tree("x"), symbol_tree("offset"))),
        Tree(MUL, (symbol_tree("x"), symbol_tree("weights"))),
        mean_tree(symbol_tree("x"), SOURCE),
        sum_tree(symbol_tree("x"), SOURCE),
        isel_tree(symbol_tree("x"), SOURCE, int_tree(0)),
    ),
)
def test_semantic_domains_match_existing_lowering(expression: Tree) -> None:
    inputs = {
        "x": domain(TARGET, SOURCE),
        "weights": domain(SOURCE),
        "offset": domain(TARGET),
    }
    described = describe_expression(expression, inputs)
    lowered = lower_expression(
        Expr(expression),
        symbol_domains=inputs,
        canonical_order=domain(TARGET, SOURCE),
    )
    assert described.domain == lowered.domain


def test_lazy_description_does_not_need_coordinate_extents() -> None:
    """A dense contraction can be described without creating O(n²) edges."""
    expression = sum_tree(
        Tree(MUL, (symbol_tree("influence"), symbol_tree("x"))),
        SOURCE,
    )
    described = describe_expression(
        expression,
        {"influence": domain(TARGET, SOURCE), "x": domain(SOURCE)},
    )
    assert described.domain == domain(TARGET)
    assert len(described.children) == len(expression.args)


def test_nested_reductions_retain_both_axis_dependencies() -> None:
    expression = sum_tree(sum_tree(symbol_tree("x"), SOURCE), TARGET)
    described = describe_expression(expression, {"x": domain(TARGET, SOURCE)})
    assert described.domain == domain()
    assert set(described.dependencies((), _registry(source=3, target=2))) == {
        SymbolCoordinate("x", (i, j)) for i in range(2) for j in range(3)
    }


def test_single_output_dependency_query_works_for_large_axis() -> None:
    """Creating the description does not enumerate the other target rows."""
    expression = sum_tree(
        Tree(MUL, (symbol_tree("matrix"), symbol_tree("x"))),
        SOURCE,
    )
    described = describe_expression(
        expression,
        {"matrix": domain(TARGET, SOURCE), "x": domain(SOURCE)},
    )
    size = 1000
    registry = _registry(source=size, target=size)
    unknowns = {dependency.indices for dependency in described.dependencies((37,), registry) if dependency.name == "x"}
    assert len(unknowns) == size
    assert (0,) in unknowns
    assert (size - 1,) in unknowns


def test_unknown_symbolic_leaf_fails_explicitly() -> None:
    with pytest.raises(BiltOperatorSemanticsError, match="Unsupported symbolic leaf"):
        describe_expression(Tree(axis(0)), {})


def test_unary_operators_require_one_operand() -> None:
    with pytest.raises(BiltOperatorSemanticsError, match="must have one operand"):
        describe_expression(
            Tree(NEG, (symbol_tree("x"), symbol_tree("y"))),
            {
                "x": domain(),
                "y": domain(),
            },
        )


def test_power_requires_two_operands() -> None:
    with pytest.raises(BiltOperatorSemanticsError, match="Power must have two operands"):
        describe_expression(Tree(POW, (symbol_tree("x"),)), {"x": domain()})


def test_coordinate_rank_is_checked_before_tracing() -> None:
    expression = describe_expression(Tree(NEG, (symbol_tree("x"),)), {"x": domain(TARGET)})
    with pytest.raises(BiltOperatorSemanticsError, match="coordinate rank"):
        list(expression.dependencies((), _registry()))
