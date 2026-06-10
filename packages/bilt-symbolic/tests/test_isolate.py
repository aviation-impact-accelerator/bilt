import pytest

from bilt.symbolic._errors import IsolationError
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolate import isolate
from bilt.symbolic._relations import IsolatedRelation, Relation


def assert_relation_reps_equal(
    relation: IsolatedRelation,
    *,
    lhs: Expr,
    rhs: Expr,
) -> None:
    assert relation.lhs.rep == lhs.rep
    assert relation.rhs.rep == rhs.rep


def test_isolate_symbol_on_lhs_identity() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(lhs=x, rhs=y)

    isolated = isolate(relation, x)

    assert_relation_reps_equal(isolated, lhs=x, rhs=y)


def test_isolate_symbol_on_rhs_swaps_sides() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(lhs=y, rhs=x)

    isolated = isolate(relation, x)

    assert_relation_reps_equal(isolated, lhs=x, rhs=y)


def test_isolate_addition() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")

    relation = Relation(lhs=x + y, rhs=z)

    isolated = isolate(relation, x)

    assert_relation_reps_equal(isolated, lhs=x, rhs=z - y)


def test_isolate_target_inside_rhs_expression() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")

    relation = Relation(lhs=z, rhs=x + y)

    isolated = isolate(relation, x)

    assert_relation_reps_equal(isolated, lhs=x, rhs=z - y)


def test_isolate_raises_if_target_absent() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")

    relation = Relation(lhs=y, rhs=z)

    with pytest.raises(IsolationError, match="does not occur"):
        isolate(relation, x)


def test_isolate_raises_if_target_occurs_more_than_once_on_one_side() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(lhs=x + x, rhs=y)

    with pytest.raises(IsolationError, match="occurs more than once"):
        isolate(relation, x)


def test_isolate_raises_if_target_occurs_on_both_sides() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = Relation(lhs=x + y, rhs=x)

    with pytest.raises(IsolationError, match="occurs more than once"):
        isolate(relation, x)


def test_isolate_rejects_non_symbol_target() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")

    relation = Relation(lhs=x + y, rhs=z)

    with pytest.raises(TypeError, match="Expected a single symbolic variable"):
        isolate(relation, x + y)
