from __future__ import annotations

import typing

import pytest

from bilt.core._equation import equation
from bilt.core._error import EquationReturnError
from bilt.symbolic._atoms import ADD, EQUALS
from bilt.symbolic._dim import Dim
from bilt.symbolic._domains import Domain, domain

# Expr is used in @equation function annotations. Those annotations are part of
# the runtime behaviour being tested, so this must stay as a runtime import.
from bilt.symbolic._expressions import Expr, symbol_expr
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


def test_equation_builds_scalar_domain_tree_by_default() -> None:
    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    assert model.__bilt_equation_tree__.domain == Domain.scalar()


def test_equation_can_be_called_with_domain_aware_symbols() -> None:
    time = Dim("time")
    time_domain = domain(time)

    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    a = symbol_expr("a", domain=time_domain)
    b = symbol_expr("b")
    c = symbol_expr("c")

    symbolic_model = typing.cast("typing.Callable[..., Tree]", model)
    tree = symbolic_model(a, b, c)

    assert tree.is_relation is True
    assert tree.domain == time_domain


def test_equation_call_with_domain_symbols_broadcasts_rhs() -> None:
    time = Dim("time")
    time_domain = domain(time)

    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    a = symbol_expr("a")
    b = symbol_expr("b", domain=time_domain)
    c = symbol_expr("c")

    symbolic_model = typing.cast("typing.Callable[..., Tree]", model)
    tree = symbolic_model(a, b, c)

    assert tree.domain == time_domain
    assert tree.right.domain == time_domain


def test_equation_call_with_symbols_merges_domains() -> None:
    time = Dim("time")
    node = Dim("node")

    @equation
    def model(a: float, b: float, c: float) -> bool:
        return a == b + c

    a = symbol_expr("a", domain=domain(time))
    b = symbol_expr("b", domain=domain(node))
    c = symbol_expr("c")

    symbolic_model = typing.cast("typing.Callable[..., Tree]", model)
    tree = symbolic_model(a, b, c)

    assert tree.domain == domain(time, node)


def test_intermediate_symbol_broadcasts_across_two_scalar_equations() -> None:
    time = Dim("time")
    node = Dim("node")

    @equation
    def first(out: float, source: float) -> bool:
        return out == source

    @equation
    def second(lhs: float, out: float, offset: float) -> bool:
        return lhs == out + offset

    assert first.__bilt_equation_tree__.domain == Domain.scalar()
    assert second.__bilt_equation_tree__.domain == Domain.scalar()

    first_symbolic = typing.cast("typing.Callable[..., Tree]", first)
    second_symbolic = typing.cast("typing.Callable[..., Tree]", second)

    out = symbol_expr("out", domain=domain(time))
    source = symbol_expr("source")

    first_tree = first_symbolic(out, source)

    assert first_tree.is_relation is True
    assert first_tree.domain == domain(time)

    lhs = symbol_expr("lhs")
    offset = symbol_expr("offset", domain=domain(node))

    second_tree = second_symbolic(lhs, out, offset)

    assert second_tree.is_relation is True
    assert second_tree.right.domain == domain(time, node)
    assert second_tree.domain == domain(time, node)


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
