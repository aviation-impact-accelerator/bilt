import typing

import pytest

from bilt.core._equation import equation
from bilt.core._error import EquationReturnError
from bilt.symbolic._atoms import ADD, EQUALS
from bilt.symbolic._expressions import Expr
from bilt.symbolic._tree import Tree


def test_equation_builds_tree_from_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert model.__bilt_equation_name__ == "model"
    assert model.__bilt_equation_parameters__ == ("a", "b", "c")
    assert tuple(model.__bilt_equation_symbols__) == ("a", "b", "c")

    tree = model.__bilt_equation_tree__

    assert isinstance(tree, Tree)
    assert tree.is_relation is True
    assert tree.is_equality is True
    assert tree.head is EQUALS


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


def test_equation_tree_uses_generated_symbols() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    symbols = model.__bilt_equation_symbols__
    tree = model.__bilt_equation_tree__

    assert tree.left is symbols["a"].rep

    rhs = tree.right

    assert rhs.head is ADD
    assert rhs.args == (symbols["b"].rep, symbols["c"].rep)


def test_equation_records_parameters_in_signature_order() -> None:
    @equation
    def model(z: float, x: float, y: float) -> bool:
        return z == x + y

    assert model.__bilt_equation_parameters__ == ("z", "x", "y")
    assert tuple(model.__bilt_equation_symbols__) == ("z", "x", "y")


def test_equation_str_includes_name_and_tree() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert str(model) == f"model: {model.__bilt_equation_tree__}"


def test_equation_call_delegates_to_original_function() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    a = 1.0
    b = 2.0
    c = 3.0

    result = model(a, b, c)

    assert result == (a == b + c)


def test_equation_raises_if_function_returns_expr() -> None:
    def bad_model(x: Expr) -> object:
        return x

    with pytest.raises(
        EquationReturnError,
        match="Did you forget to compare it with another expression",
    ):
        equation(bad_model)


def test_equation_raises_if_function_returns_non_relation_tree() -> None:
    def bad_model(x: Expr) -> object:
        return x.rep

    with pytest.raises(EquationReturnError, match="non-relation tree"):
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
