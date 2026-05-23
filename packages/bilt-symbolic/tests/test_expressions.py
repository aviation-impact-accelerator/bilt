import pytest

from bilt.symbolic import (
    Binary,
    BinaryOperator,
    Constant,
    Symbol,
    add,
    as_expr,
    contains,
    div,
    evaluate,
    mul,
    parse_expr,
    pow_,
    sub,
    symbols,
)


def test_as_expr_returns_existing_expr() -> None:
    expr = Symbol("x")

    result = as_expr(expr)

    assert result is expr


def test_as_expr_converts_number_to_constant() -> None:
    value = 3.5

    result = as_expr(value)

    assert isinstance(result, Constant)
    assert result.value == value


def test_constant_eval_returns_value() -> None:
    value = 2.5
    expr = Constant(value)

    result = evaluate(expr, {})

    assert result == value


def test_constant_has_no_symbols() -> None:
    expr = Constant(2.0)

    assert symbols(expr) == set()
    assert not contains(expr, "x")


def test_constant_str_returns_value() -> None:
    expr = Constant(2.0)

    assert str(expr) == "2.0"


def test_symbol_eval_returns_value_from_environment() -> None:
    value = 3.0
    expr = Symbol("x")

    result = evaluate(expr, {"x": value})

    assert result == value


def test_symbol_eval_raises_for_missing_value() -> None:
    expr = Symbol("x")

    with pytest.raises(KeyError, match="Missing value for symbol 'x'"):
        evaluate(expr, {})


def test_symbol_reports_its_own_symbol() -> None:
    expr = Symbol("x")

    assert symbols(expr) == {"x"}
    assert contains(expr, "x")
    assert not contains(expr, "y")


def test_symbol_str_uses_name_by_default() -> None:
    expr = Symbol("x")

    assert str(expr) == "x"


def test_symbol_str_uses_display_name_if_provided() -> None:
    expr = Symbol("density", display_name="rho")

    assert str(expr) == "rho"


@pytest.mark.parametrize(
    ("op", "expected"),
    (
        ("+", 5.0),
        ("-", 1.0),
        ("*", 6.0),
        ("/", 1.5),
        ("**", 9.0),
    ),
)
def test_binary_eval(op: BinaryOperator, expected: float) -> None:
    expr = Binary(op, Constant(3.0), Constant(2.0))

    result = evaluate(expr, {})

    assert result == expected


def test_binary_eval_with_symbols() -> None:
    expected = 20.0
    expr = Binary("*", Binary("+", Symbol("x"), Constant(2.0)), Symbol("y"))

    result = evaluate(expr, {"x": 3.0, "y": 4.0})

    assert result == expected


def test_parse_expr_rejects_unknown_binary_operator() -> None:
    with pytest.raises(SyntaxError, match="Unsupported Python binary operator"):
        parse_expr("x % y")


def test_binary_symbols_combines_symbols_from_both_sides() -> None:
    expr = Binary("+", Symbol("x"), Binary("*", Symbol("y"), Constant(2.0)))

    assert symbols(expr) == {"x", "y"}


def test_binary_contains_checks_both_sides() -> None:
    expr = Binary("+", Symbol("x"), Binary("*", Symbol("y"), Constant(2.0)))

    assert contains(expr, "x")
    assert contains(expr, "y")
    assert not contains(expr, "z")


def test_binary_str() -> None:
    expr = Binary("+", Symbol("x"), Constant(2.0))

    assert str(expr) == "(x + 2.0)"


def test_add_creates_binary_expression() -> None:
    expected = 5.0
    x = Symbol("x")

    expr = add(x, 2.0)

    assert isinstance(expr, Binary)
    assert expr.op == "+"
    assert expr.left == x
    assert expr.right == Constant(2.0)
    assert evaluate(expr, {"x": 3.0}) == expected


def test_reverse_add_operator_creates_binary_expression() -> None:
    expected = 5.0
    x = Symbol("x")

    expr = add(2.0, x)

    assert isinstance(expr, Binary)
    assert expr.op == "+"
    assert expr.left == Constant(2.0)
    assert expr.right == x
    assert evaluate(expr, {"x": 3.0}) == expected


def test_sub_operator_preserves_left_and_right_order() -> None:
    expected = 1.0
    x = Symbol("x")

    expr = sub(x, 2.0)

    assert isinstance(expr, Binary)
    assert expr.op == "-"
    assert expr.left == x
    assert expr.right == Constant(2.0)
    assert evaluate(expr, {"x": 3.0}) == expected


def test_reverse_sub_operator_preserves_left_and_right_order() -> None:
    expected = 7.0
    x = Symbol("x")

    expr = sub(10.0, x)

    assert isinstance(expr, Binary)
    assert expr.op == "-"
    assert expr.left == Constant(10.0)
    assert expr.right == x
    assert evaluate(expr, {"x": 3.0}) == expected


def test_mul_operator_creates_binary_expression() -> None:
    expected = 6.0
    x = Symbol("x")

    expr = mul(x, 2.0)

    assert isinstance(expr, Binary)
    assert expr.op == "*"
    assert expr.left == x
    assert expr.right == Constant(2.0)
    assert evaluate(expr, {"x": 3.0}) == expected


def test_reverse_mul_operator_creates_binary_expression() -> None:
    expected = 6.0
    x = Symbol("x")

    expr = mul(2.0, x)

    assert isinstance(expr, Binary)
    assert expr.op == "*"
    assert expr.left == Constant(2.0)
    assert expr.right == x
    assert evaluate(expr, {"x": 3.0}) == expected


def test_div_operator_preserves_left_and_right_order() -> None:
    expected = 1.5
    x = Symbol("x")

    expr = div(x, 2.0)

    assert isinstance(expr, Binary)
    assert expr.op == "/"
    assert expr.left == x
    assert expr.right == Constant(2.0)
    assert evaluate(expr, {"x": 3.0}) == expected


def test_reverse_div_operator_preserves_left_and_right_order() -> None:
    expected = 4.0
    x = Symbol("x")

    expr = div(12.0, x)

    assert isinstance(expr, Binary)
    assert expr.op == "/"
    assert expr.left == Constant(12.0)
    assert expr.right == x
    assert evaluate(expr, {"x": 3.0}) == expected


def test_pow_operator_preserves_left_and_right_order() -> None:
    expected = 9.0
    x = Symbol("x")

    expr = pow_(x, 2.0)

    assert isinstance(expr, Binary)
    assert expr.op == "**"
    assert expr.left == x
    assert expr.right == Constant(2.0)
    assert evaluate(expr, {"x": 3.0}) == expected


def test_reverse_pow_operator_preserves_left_and_right_order() -> None:
    expected = 8.0
    x = Symbol("x")

    expr = pow_(2.0, x)

    assert isinstance(expr, Binary)
    assert expr.op == "**"
    assert expr.left == Constant(2.0)
    assert expr.right == x
    assert evaluate(expr, {"x": 3.0}) == expected


def test_expression_evaluation_and_symbol_inspection() -> None:
    expected = 11.0
    expr = Binary("+", Symbol("x"), Binary("*", Symbol("y"), Constant(2.0)))

    result = evaluate(expr, {"x": 3.0, "y": 4.0})

    assert result == expected
    assert symbols(expr) == {"x", "y"}
    assert contains(expr, "x")
    assert contains(expr, "y")
    assert not contains(expr, "z")
