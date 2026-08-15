import pytest

from bilt.symbolic._algebra._expr_ops import (
    make_add,
    make_mul,
    make_neg,
    same_tree,
)
from bilt.symbolic._algebra._linearise import (
    LinearisationError,
    linearise,
)
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import Tree, int_tree


def assert_same_tree(left: Tree, right: Tree) -> None:
    assert same_tree(left, right)


def test_linearise() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    a = Expr.symbol("a")
    b = Expr.symbol("b")

    system = linearise(
        (
            make_add(x, y) == a,
            make_add(x, make_neg(y)) == b,
        ),
        ("x", "y"),
    )

    assert system.unknowns == ("x", "y")
    assert system.parameters == ("a", "b")

    assert_same_tree(system.coefficients[0][0], int_tree(1))
    assert_same_tree(system.coefficients[0][1], int_tree(1))
    assert_same_tree(system.coefficients[1][0], int_tree(1))
    assert_same_tree(system.coefficients[1][1], make_neg(Expr(int_tree(1))).rep)

    assert_same_tree(system.rhs[0], a.rep)
    assert_same_tree(system.rhs[1], b.rep)


def test_linearise_symbolic_coefficients() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    b = Expr.symbol("b")

    system = linearise(
        (make_mul(a, x) == b,),
        ("x",),
    )

    assert system.parameters == ("a", "b")
    assert_same_tree(system.coefficients[0][0], a.rep)
    assert_same_tree(system.rhs[0], b.rep)


def test_linearise_rejects_nonlinear_equation() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    with pytest.raises(LinearisationError):
        linearise(
            (make_mul(x, x) == y,),
            ("x",),
        )


def test_linearise_requires_square_system() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    with pytest.raises(LinearisationError):
        linearise(
            (x == y,),
            ("x", "y"),
        )
