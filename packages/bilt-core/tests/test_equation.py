import typing

import pytest

from bilt.core._equation import equation
from bilt.core._error import EquationReturnError
from bilt.symbolic._expressions import Expr
from bilt.symbolic._relations import Relation, equals, relation_to_string


def test_equals_returns_relation() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    rel = equals(x, y)

    assert isinstance(rel, Relation)
    assert rel.lhs is x
    assert rel.rhs is y


def test_relation_to_string_formats_equality() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    rel = equals(x, y)

    assert relation_to_string(rel) == f"{x} == {y}"


def test_relation_str_uses_relation_to_string() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    rel = equals(x, y)

    assert str(rel) == relation_to_string(rel)


def test_equation_builds_equation_from_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert model.__bilt_equation_name__ == "model"
    assert model.__bilt_equation_parameters__ == ("a", "b", "c")
    assert set(model.__bilt_equation_symbols__) == {"a", "b", "c"}
    assert isinstance(model.__bilt_equation_relation__, Relation)


def test_equation_passes_expr_symbols_to_function() -> None:
    seen: dict[str, Expr] = {}

    @equation
    def model(a: float, b: float, c: float) -> bool:
        seen["a"] = typing.cast("Expr", a)
        seen["b"] = typing.cast("Expr", b)
        seen["c"] = typing.cast("Expr", c)
        return a == b + c

    assert model.__bilt_equation_symbols__["a"] is seen["a"]
    assert model.__bilt_equation_symbols__["b"] is seen["b"]
    assert model.__bilt_equation_symbols__["c"] is seen["c"]


def test_equation_relation_uses_generated_symbols() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    symbols = model.__bilt_equation_symbols__
    relation = model.__bilt_equation_relation__

    assert relation.lhs is symbols["a"]
    assert str(relation.rhs) == str(symbols["b"] + symbols["c"])


def test_equation_records_parameters_in_signature_order() -> None:
    @equation
    def model(z: float, x: float, y: float) -> bool:
        return z == x + y

    assert model.__bilt_equation_parameters__ == ("z", "x", "y")
    assert tuple(model.__bilt_equation_symbols__) == ("z", "x", "y")


def test_equation_str_includes_name_and_relation() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert str(model) == f"model: {model.__bilt_equation_relation__}"


def test_equation_call_delegates_to_original_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    rel = model(1.0, 2.0, 3.0)

    assert rel == (1.0 == 2.0 + 3.0)


def test_equation_raises_if_function_does_not_return_relation() -> None:
    def bad_model(x: Expr) -> object:
        return x

    with pytest.raises(EquationReturnError, match="Did you forget to compare it with another expression"):
        equation(bad_model)


def test_equation_error_mentions_function_name() -> None:
    def bad_model(x: Expr) -> object:
        return x

    with pytest.raises(EquationReturnError, match="'bad_model'"):
        equation(bad_model)


def test_equation_error_mentions_return_type() -> None:
    def bad_model(_x: Expr) -> object:
        return 1

    with pytest.raises(EquationReturnError, match="int"):
        equation(bad_model)
