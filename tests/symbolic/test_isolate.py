import typing

import pytest

from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import (
    TargetNotFoundError,
    TargetOnBothSidesError,
    UnsupportedIsolationError,
)
from bilt.symbolic._isolation._expr_ops import (
    make_add,
    make_inv,
    make_mul,
    make_neg,
    same_tree,
)
from bilt.symbolic._isolation._isolate import isolate

if typing.TYPE_CHECKING:
    from bilt.symbolic._tree import Tree


def assert_same_tree(left: Tree, right: Tree) -> None:
    assert same_tree(left, right)


def test_isolate_already_isolated() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    actual = isolate(x == y, x)

    assert_same_tree(actual.as_tree(), x == y)


def test_isolate_swaps_sides_when_target_is_rhs() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    actual = isolate(y == x, x)

    assert_same_tree(actual.as_tree(), x == y)


def test_isolate_additive_term() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    actual = isolate(make_add(x, a) == y, x)

    expected_rhs = make_add(y, make_neg(a))
    assert_same_tree(actual.as_tree(), x == expected_rhs)


def test_isolate_additive_term_after_swapping_sides() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    actual = isolate(y == make_add(x, a), x)

    expected_rhs = make_add(y, make_neg(a))
    assert_same_tree(actual.as_tree(), x == expected_rhs)


def test_isolate_multiplicative_factor() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    actual = isolate(make_mul(a, x) == y, x)

    expected_rhs = make_mul(y, make_inv(a))
    assert_same_tree(actual.as_tree(), x == expected_rhs)


def test_isolate_multiplicative_factor_after_swapping_sides() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    actual = isolate(y == make_mul(a, x), x)

    expected_rhs = make_mul(y, make_inv(a))
    assert_same_tree(actual.as_tree(), x == expected_rhs)


def test_isolate_nested_add_inside_mul() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    b = Expr.symbol("b")
    y = Expr.symbol("y")

    actual = isolate(make_mul(a, make_add(x, b)) == y, x)

    expected_rhs = make_add(make_mul(y, make_inv(a)), make_neg(b))
    assert_same_tree(actual.as_tree(), x == expected_rhs)


def test_isolate_nested_add_inside_mul_after_swapping_sides() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    b = Expr.symbol("b")
    y = Expr.symbol("y")

    actual = isolate(y == make_mul(a, make_add(x, b)), x)

    expected_rhs = make_add(make_mul(y, make_inv(a)), make_neg(b))
    assert_same_tree(actual.as_tree(), x == expected_rhs)


def test_isolate_raises_when_target_missing() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    with pytest.raises(TargetNotFoundError):
        isolate(a == y, x)


def test_isolate_raises_when_target_on_both_sides() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")

    with pytest.raises(TargetOnBothSidesError):
        isolate(make_add(x, a) == x, x)


def test_isolate_raises_when_target_occurs_more_than_once() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    with pytest.raises(UnsupportedIsolationError):
        isolate(make_add(x, x) == y, x)
