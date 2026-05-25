import pytest

from bilt.symbolic._expressions import Constant, Symbol, add, sub
from bilt.symbolic._relations import (
    Eq,
    Ge,
    Le,
    Relation,
    eq,
    evaluate_relation,
    ge,
    le,
    relation,
    relation_kind,
    relation_symbols,
    relation_to_string,
    residual,
)


def test_relation_constructor_coerces_int_and_float_to_constants() -> None:
    rel = relation("==", 1, 2.5)

    assert rel == Relation("==", Constant(1.0), Constant(2.5))


def test_relation_constructor_accepts_expressions() -> None:
    x = Symbol("x")
    y = Symbol("y")

    rel = relation("==", x, y)

    assert rel == Relation("==", x, y)


def test_relation_constructor_rejects_bool() -> None:
    with pytest.raises(TypeError, match="Boolean values are not supported"):
        relation("==", True, 1.0)  # type: ignore[arg-type]


def test_relation_constructor_rejects_unsupported_value() -> None:
    with pytest.raises(TypeError, match="Unsupported expression value"):
        relation("==", "x", 1.0)  # type: ignore[arg-type]


def test_eq_constructor() -> None:
    x = Symbol("x")

    rel = eq(x, 1.0)

    assert rel == Relation("==", x, Constant(1.0))


def test_le_constructor() -> None:
    x = Symbol("x")

    rel = le(x, 1.0)

    assert rel == Relation("<=", x, Constant(1.0))


def test_ge_constructor() -> None:
    x = Symbol("x")

    rel = ge(x, 1.0)

    assert rel == Relation(">=", x, Constant(1.0))


def test_capitalised_constructor_aliases() -> None:
    x = Symbol("x")

    assert Eq(x, 1.0) == eq(x, 1.0)
    assert Le(x, 1.0) == le(x, 1.0)
    assert Ge(x, 1.0) == ge(x, 1.0)


def test_relation_to_string_for_equality() -> None:
    x = Symbol("x")
    rel = Eq(x, 1.0)

    assert relation_to_string(rel) == "x == 1.0"


def test_relation_to_string_for_less_equal() -> None:
    x = Symbol("x")
    rel = Le(x, 1.0)

    assert relation_to_string(rel) == "x <= 1.0"


def test_relation_to_string_for_greater_equal() -> None:
    x = Symbol("x")
    rel = Ge(x, 1.0)

    assert relation_to_string(rel) == "x >= 1.0"


def test_relation_str_uses_relation_to_string() -> None:
    x = Symbol("x")
    rel = Eq(x, 1.0)

    assert str(rel) == "x == 1.0"


def test_relation_kind_for_equality() -> None:
    x = Symbol("x")

    assert relation_kind(Eq(x, 1.0)) == "equality"


def test_relation_kind_for_less_equal() -> None:
    x = Symbol("x")

    assert relation_kind(Le(x, 1.0)) == "less_equal"


def test_relation_kind_for_greater_equal() -> None:
    x = Symbol("x")

    assert relation_kind(Ge(x, 1.0)) == "greater_equal"


def test_residual_returns_lhs_minus_rhs() -> None:
    x = Symbol("x")
    y = Symbol("y")
    rel = Eq(add(x, y), 10.0)

    result = residual(rel)

    assert result == sub(add(x, y), Constant(10.0))


def test_residual_is_same_for_inequalities() -> None:
    x = Symbol("x")

    assert residual(Le(x, 5.0)) == sub(x, Constant(5.0))
    assert residual(Ge(x, 5.0)) == sub(x, Constant(5.0))


def test_relation_symbols_for_constants() -> None:
    rel = Eq(1.0, 2.0)

    assert relation_symbols(rel) == set()


def test_relation_symbols_for_single_symbol() -> None:
    x = Symbol("x")
    rel = Eq(x, 2.0)

    assert relation_symbols(rel) == {"x"}


def test_relation_symbols_for_multiple_symbols() -> None:
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")

    rel = Eq(add(x, y), z)

    assert relation_symbols(rel) == {"x", "y", "z"}


def test_evaluate_relation_equality_true() -> None:
    x = Symbol("x")
    rel = Eq(x, 3.0)

    assert evaluate_relation(rel, {"x": 3.0}) is True


def test_evaluate_relation_equality_false() -> None:
    x = Symbol("x")
    rel = Eq(x, 3.0)

    assert evaluate_relation(rel, {"x": 4.0}) is False


def test_evaluate_relation_less_equal_true() -> None:
    x = Symbol("x")
    rel = Le(x, 3.0)

    assert evaluate_relation(rel, {"x": 2.0}) is True


def test_evaluate_relation_less_equal_equal_boundary() -> None:
    x = Symbol("x")
    rel = Le(x, 3.0)

    assert evaluate_relation(rel, {"x": 3.0}) is True


def test_evaluate_relation_less_equal_false() -> None:
    x = Symbol("x")
    rel = Le(x, 3.0)

    assert evaluate_relation(rel, {"x": 4.0}) is False


def test_evaluate_relation_greater_equal_true() -> None:
    x = Symbol("x")
    rel = Ge(x, 3.0)

    assert evaluate_relation(rel, {"x": 4.0}) is True


def test_evaluate_relation_greater_equal_equal_boundary() -> None:
    x = Symbol("x")
    rel = Ge(x, 3.0)

    assert evaluate_relation(rel, {"x": 3.0}) is True


def test_evaluate_relation_greater_equal_false() -> None:
    x = Symbol("x")
    rel = Ge(x, 3.0)

    assert evaluate_relation(rel, {"x": 2.0}) is False


def test_evaluate_relation_with_expression_sides() -> None:
    x = Symbol("x")
    y = Symbol("y")

    rel = Eq(add(x, y), 10.0)

    assert evaluate_relation(rel, {"x": 4.0, "y": 6.0}) is True


def test_evaluate_relation_raises_for_missing_symbol() -> None:
    x = Symbol("x")
    rel = Eq(x, 1.0)

    with pytest.raises(KeyError, match="Missing value for symbol 'x'"):
        evaluate_relation(rel, {})