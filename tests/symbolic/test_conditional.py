import pytest

from bilt.core._dimensional_analysis import absoluteness_of, dimensionality_of
from bilt.core._port_declaration import PortDeclaration
from bilt.errors import (
    BiltDimensionalityError,
    BiltEquationDimensionalityError,
    BiltSymbolicConversionError,
    BiltUnsupportedIsolationError,
)
from bilt.symbolic._atoms import CONDITIONAL
from bilt.symbolic._expressions import Expr, conditional_expr, tensor_symbol_expr
from bilt.symbolic._isolation._isolate import isolate
from bilt.symbolic._tree import conditional_tree
from bilt.symbolic._walk import contains
from bilt.units.dimensions import LENGTH, MASS, TEMPERATURE


def test_conditional_has_three_operands_and_is_not_a_relation() -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    tree = conditional_expr(x > 0, a, b).tree
    assert tree.head is CONDITIONAL
    assert tree.args == (x > 0, a.tree, b.tree)
    assert not tree.is_relation


def test_conditional_requires_a_relation_predicate() -> None:
    x = Expr.symbol("x")
    with pytest.raises(TypeError, match="binary relation predicate"):
        conditional_tree(x.tree, x.tree, x.tree)


def test_conditional_rejects_relation_valued_branches() -> None:
    x, y = Expr.symbol("x"), Expr.symbol("y")
    with pytest.raises(TypeError, match="branches must be expressions"):
        conditional_tree(x > 0, x == y, y.tree)


@pytest.mark.parametrize("tensor_first", (True, False))
def test_tensor_conditional_branches_are_deferred(tensor_first: bool) -> None:
    x = Expr.symbol("x")
    tensor = tensor_symbol_expr("tensor")
    a, b = (tensor, x) if tensor_first else (x, tensor)
    with pytest.raises(BiltSymbolicConversionError, match="Tensor-valued conditional branches"):
        conditional_expr(x > 0, a, b)


def test_dependencies_include_predicate_and_both_branches() -> None:
    selector, a, b = (Expr.symbol(name) for name in ("selector", "a", "b"))
    tree = conditional_expr(selector > 0, a, b).tree
    for expr in (selector, a, b):
        assert contains(tree, expr.tree)


def test_conditional_preserves_branch_dimensionality() -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    tree = conditional_expr(x > 0, a, b).tree
    assert dimensionality_of(tree, {"x": MASS, "a": LENGTH, "b": LENGTH}) is LENGTH


def test_conditional_rejects_different_branch_dimensions() -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    tree = conditional_expr(x > 0, a, b).tree
    with pytest.raises(BiltDimensionalityError):
        dimensionality_of(tree, {"x": MASS, "a": LENGTH, "b": MASS})


def test_conditional_validates_predicate_dimensionality() -> None:
    x, threshold, a = (Expr.symbol(name) for name in ("x", "threshold", "a"))
    tree = conditional_expr(x > threshold, a, a).tree
    with pytest.raises(BiltDimensionalityError):
        dimensionality_of(tree, {"x": MASS, "threshold": LENGTH, "a": LENGTH})


def test_conditional_rejects_different_branch_kinds() -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    declarations = {
        "a": PortDeclaration(TEMPERATURE, absolute=True),
        "b": PortDeclaration(TEMPERATURE),
    }
    with pytest.raises(BiltEquationDimensionalityError, match="conditional branches"):
        absoluteness_of("relation", conditional_expr(x > 0, a, b).tree, declarations)


def test_conditional_preserves_absolute_branch_kind() -> None:
    x, a, b = (Expr.symbol(name) for name in ("x", "a", "b"))
    declarations = {
        "a": PortDeclaration(TEMPERATURE, absolute=True),
        "b": PortDeclaration(TEMPERATURE, absolute=True),
    }
    assert absoluteness_of("relation", conditional_expr(x > 0, a, b).tree, declarations) == {1}


@pytest.mark.parametrize("target_on_lhs", (True, False))
def test_isolation_rearranges_around_independent_conditional(target_on_lhs: bool) -> None:
    x, y, a, b, selector = (Expr.symbol(name) for name in ("x", "y", "a", "b", "selector"))
    selected = conditional_expr(selector > 0, a, b)
    relation = x + selected == y if target_on_lhs else y == x + selected

    result = isolate(relation, x)
    assert not contains(result.rhs.tree, x.tree)
    assert contains(result.rhs.tree, selected.tree)


@pytest.mark.parametrize("location", ("predicate", "true", "false"))
def test_isolation_rejects_inversion_through_conditional(location: str) -> None:
    x, y, selector = (Expr.symbol(name) for name in ("x", "y", "selector"))
    predicate = (x if location == "predicate" else selector) > 0
    when_true = x if location == "true" else y
    when_false = x if location == "false" else y
    selected = conditional_expr(predicate, when_true, when_false)
    with pytest.raises(BiltUnsupportedIsolationError, match="through a conditional"):
        isolate(selected == y, x)


def test_isolation_rejects_already_isolated_target_with_conditional_dependency() -> None:
    x, y, z = (Expr.symbol(name) for name in ("x", "y", "z"))
    selected = conditional_expr(x > 0, y, z)
    with pytest.raises(BiltUnsupportedIsolationError, match="through a conditional"):
        isolate(x == selected, x)
