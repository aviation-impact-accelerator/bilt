import pytest

from bilt.core._equation import equation
from bilt.symbolic._equalities import Equality
from bilt.symbolic._expressions import Expr
from bilt.symbolic._relations import Relation, as_relation, equals


def assert_same_expr(left: Expr, right: Expr) -> None:
    assert left.rep is right.rep


def test_expr_eq_builds_equality() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    equality = x == y

    assert isinstance(equality, Equality)
    assert equality.lhs is x
    assert equality.rhs is y


def test_equals_stores_lhs_and_rhs_exprs() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    rel = equals(x, y)

    assert rel.lhs is x
    assert rel.rhs is y


def test_equals_coerces_int_lhs() -> None:
    y = Expr.symbol("y")

    rel = equals(1, y)

    assert_same_expr(rel.lhs, Expr.integer(1))
    assert rel.rhs is y


def test_equals_coerces_int_rhs() -> None:
    x = Expr.symbol("x")

    rel = equals(x, 1)

    assert rel.lhs is x
    assert_same_expr(rel.rhs, Expr.integer(1))


def test_equals_coerces_float_lhs() -> None:
    y = Expr.symbol("y")

    rel = equals(1.5, y)

    assert_same_expr(rel.lhs, Expr.real(1.5))
    assert rel.rhs is y


def test_equals_coerces_float_rhs() -> None:
    x = Expr.symbol("x")

    rel = equals(x, 1.5)

    assert rel.lhs is x
    assert_same_expr(rel.rhs, Expr.real(1.5))


def test_equals_rejects_bool_lhs() -> None:
    y = Expr.symbol("y")
    value = True

    with pytest.raises(TypeError, match="Cannot convert bool to Expr"):
        equals(value, y)


def test_equals_rejects_bool_rhs() -> None:
    x = Expr.symbol("x")
    value = False

    with pytest.raises(TypeError, match="Cannot convert bool to Expr"):
        equals(x, value)


def test_equals_rejects_unsupported_lhs() -> None:
    y = Expr.symbol("y")

    with pytest.raises(TypeError, match="Cannot convert object to Expr"):
        equals(object(), y)


def test_equals_rejects_unsupported_rhs() -> None:
    x = Expr.symbol("x")

    with pytest.raises(TypeError, match="Cannot convert object to Expr"):
        equals(x, object())


def test_relation_is_frozen() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")
    rel = Relation(lhs=x, rhs=y)
    with pytest.raises(AttributeError):
        setattr(rel, "lhs", z)  # noqa: B010


def test_relation_uses_slots() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    rel = Relation(lhs=x, rhs=y)

    assert not hasattr(rel, "__dict__")


def test_as_relation_converts_equality() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    equality = equals(x, y)
    relation = as_relation(equality)

    assert isinstance(relation, Relation)
    assert relation.lhs is x
    assert relation.rhs is y


def test_as_relation_returns_relation_unchanged() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    relation = Relation(lhs=x, rhs=y)

    assert as_relation(relation) is relation


def test_equation_builds_equation_from_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> object:
        return a == b + c

    assert model.__bilt_equation_name__ == "model"
    assert model.__bilt_equation_parameters__ == ("a", "b", "c")
    assert set(model.__bilt_equation_symbols__) == {"a", "b", "c"}
    assert isinstance(model.__bilt_equation_relation__, Relation)


def test_equation_accepts_equals_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> object:
        return equals(a, b + c)

    assert isinstance(model.__bilt_equation_relation__, Relation)
