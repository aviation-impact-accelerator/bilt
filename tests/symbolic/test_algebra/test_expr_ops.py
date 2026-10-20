from bilt.symbolic._algebra._expr_ops import (
    add,
    divide,
    is_one,
    is_zero,
    multiply,
    negate,
    one,
    same_tree,
    subtract,
    zero,
)
from bilt.symbolic._expressions import Expr


def test_zero_and_one() -> None:
    assert is_zero(zero())
    assert is_one(one())


def test_arithmetic_identities() -> None:
    x = Expr.symbol("x").rep

    assert same_tree(add(x, zero()), x)
    assert same_tree(subtract(x, zero()), x)
    assert same_tree(multiply(x, one()), x)
    assert is_zero(multiply(x, zero()))
    assert same_tree(divide(x, one()), x)


def test_double_negation_simplifies() -> None:
    x = Expr.symbol("x").rep

    assert same_tree(
        negate(negate(x)),
        x,
    )


def test_division_by_minus_one_negates() -> None:
    x = Expr.symbol("x").rep

    assert same_tree(
        divide(x, negate(one())),
        negate(x),
    )
