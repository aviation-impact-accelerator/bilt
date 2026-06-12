import pytest

from bilt.symbolic._expressions import Expr
from bilt.symbolic._relations import Relation
from bilt.symbolic._isolate_skeleton import (
    TargetNotFoundError,
    TargetOnBothSidesError,
    UnsupportedIsolationError,
    isolate,
    make_add,
    make_div,
    make_inv,
    make_mul,
    make_neg,
    make_sub,
)


def assert_same_expr(left: Expr, right: Expr) -> None:
    assert same_tree(left.rep, right.rep)


def same_tree(left, right) -> bool:
    if left is right:
        return True

    if left.head != right.head:
        return False

    if len(left.args) != len(right.args):
        return False

    return all(
        same_tree(left_child, right_child)
        for left_child, right_child in zip(left.args, right.args, strict=True)
    )


def assert_isolated(result, target: Expr, rhs: Expr) -> None:
    assert_same_expr(result.lhs, target)
    assert_same_expr(result.rhs, rhs)


def test_isolate_already_isolated() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    result = isolate(Relation(lhs=x, rhs=y), x)

    assert_isolated(result, x, y)


def test_isolate_swaps_sides_when_target_on_rhs() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    result = isolate(Relation(lhs=y, rhs=x), x)

    assert_isolated(result, x, y)


def test_isolate_additive_term() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_add(x, a),
        rhs=y,
    )

    result = isolate(relation, x)

    assert_isolated(result, x, make_sub(y, a))


def test_isolate_multiplicative_factor() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_mul(a, x),
        rhs=y,
    )

    result = isolate(relation, x)

    assert_isolated(result, x, make_div(y, a))


def test_isolate_nested_linear_expression() -> None:
    x = Expr.symbol("x")
    a = Expr.symbol("a")
    b = Expr.symbol("b")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_add(make_mul(a, x), b),
        rhs=y,
    )

    result = isolate(relation, x)

    assert_isolated(
        result,
        x,
        make_div(make_sub(y, b), a),
    )


def test_isolate_negated_target() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_neg(x),
        rhs=y,
    )

    result = isolate(relation, x)

    assert_isolated(result, x, make_neg(y))


def test_isolate_inverted_target() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_inv(x),
        rhs=y,
    )

    result = isolate(relation, x)

    assert_isolated(result, x, make_inv(y))


def test_isolate_target_missing_raises() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")

    relation = Relation(lhs=y, rhs=z)

    with pytest.raises(TargetNotFoundError):
        isolate(relation, x)


def test_isolate_target_on_both_sides_raises() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_add(x, y),
        rhs=x,
    )

    with pytest.raises(TargetOnBothSidesError):
        isolate(relation, x)


def test_isolate_repeated_target_on_one_side_raises() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(
        lhs=make_add(x, x),
        rhs=y,
    )

    with pytest.raises(UnsupportedIsolationError):
        isolate(relation, x)