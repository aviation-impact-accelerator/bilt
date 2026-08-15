import pytest

from bilt.symbolic._algebra._affine import (
    AffineAnalysisError,
    as_affine,
)
from bilt.symbolic._algebra._expr_ops import (
    add,
    multiply,
    negate,
    one,
    same_tree,
    zero,
)
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import int_tree


def test_as_affine_constant() -> None:
    a = Expr.symbol("a").rep

    form = as_affine(
        a,
        frozenset({"x"}),
    )

    assert form.coefficients == {}
    assert same_tree(form.constant, a)


def test_as_affine_symbol() -> None:
    x = Expr.symbol("x").rep

    form = as_affine(
        x,
        frozenset({"x"}),
    )

    assert form.coefficients.keys() == {"x"}
    assert same_tree(form.coefficients["x"], one())
    assert same_tree(form.constant, zero())


def test_as_affine_multiple_symbols() -> None:
    x = Expr.symbol("x").rep
    y = Expr.symbol("y").rep
    a = Expr.symbol("a").rep
    b = Expr.symbol("b").rep

    tree = add(
        add(
            multiply(int_tree(3), x),
            multiply(a, y),
        ),
        negate(b),
    )

    form = as_affine(
        tree,
        frozenset({"x", "y"}),
    )

    assert form.coefficients.keys() == {"x", "y"}
    assert same_tree(form.coefficients["x"], int_tree(3))
    assert same_tree(form.coefficients["y"], a)
    assert same_tree(form.constant, negate(b))


def test_as_affine_rejects_nonlinear_expression() -> None:
    x = Expr.symbol("x").rep
    y = Expr.symbol("y").rep

    with pytest.raises(AffineAnalysisError):
        as_affine(
            multiply(x, y),
            frozenset({"x", "y"}),
        )
