import pytest

from bilt.symbolic import contains, contains_symbols, evaluate, parse
from bilt.symbolic._expressions import (
    Constant,
    Symbol,
    Binary,
    contains_expression,
    evaluate_expression,
    expression_symbols,
    parse_expression,
)


def test_parse_returns_expr() -> None:
    expr = parse("x + 2")

    assert isinstance(expr, Constant|Symbol|Binary)


def test_parse_delegates_to_parse_expression() -> None:
    expr = parse("x + 2")
    expected = parse_expression("x + 2")

    assert expr == expected


def test_evaluate_expression_with_single_symbol() -> None:
    expr = parse("x + 2")

    result = evaluate(expr, {"x": 3.0})

    assert result == 5.0


def test_evaluate_expression_with_multiple_symbols() -> None:
    expr = parse("x * y + 2")

    result = evaluate(expr, {"x": 3.0, "y": 4.0})

    assert result == 14.0


def test_evaluate_delegates_to_evaluate_expression() -> None:
    expr = parse("x + 2")
    values = {"x": 3.0}

    assert evaluate(expr, values) == evaluate_expression(expr, values)


def test_contains_symbol_true() -> None:
    expr = parse("x + 2")

    assert contains(expr, "x") is True


def test_contains_symbol_false() -> None:
    expr = parse("x + 2")

    assert contains(expr, "y") is False


def test_contains_delegates_to_contains_expression() -> None:
    expr = parse("x + y")

    assert contains(expr, "x") == contains_expression(expr, "x")
    assert contains(expr, "z") == contains_expression(expr, "z")


def test_contains_symbols_returns_all_symbols() -> None:
    expr = parse("x * y + z")

    assert contains_symbols(expr) == {"x", "y", "z"}


def test_contains_symbols_delegates_to_expression_symbols() -> None:
    expr = parse("x * y + z")

    assert contains_symbols(expr) == expression_symbols(expr)


def test_contains_symbols_does_not_recurse_forever() -> None:
    expr = parse("x + 2")

    assert contains_symbols(expr) == {"x"}


def test_evaluate_raises_for_missing_symbol() -> None:
    expr = parse("x + 2")

    with pytest.raises(KeyError):
        evaluate(expr, {})