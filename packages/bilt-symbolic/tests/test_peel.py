import pytest

from bilt.symbolic._errors import IsolationError
from bilt.symbolic._expressions import Expr
from bilt.symbolic._peel import BinaryArgs, SymbolTarget, peel


def test_binary_args_accepts_two_args() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    args = BinaryArgs.from_args((x.rep, y.rep), "addition")

    assert args.left is x.rep
    assert args.right is y.rep


def test_binary_args_rejects_wrong_arity() -> None:
    x = Expr.symbol("x")

    with pytest.raises(IsolationError, match="Expected binary addition, got 1 arguments"):
        BinaryArgs.from_args((x.rep,), "addition")


def test_peel_returns_rhs_when_lhs_is_target() -> None:
    x = Expr.symbol("x")
    c = Expr.symbol("c")

    result = peel(lhs=x, rhs=c, target=SymbolTarget(x))

    assert result.rep == c.rep


def test_peel_add_target_on_left() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    c = Expr.symbol("c")

    result = peel(lhs=x + y, rhs=c, target=SymbolTarget(x))

    expected = c + (-y)
    assert result.rep == expected.rep


def test_peel_add_target_on_right() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    c = Expr.symbol("c")

    result = peel(lhs=y + x, rhs=c, target=SymbolTarget(x))
    expected = c + (-y)
    assert result.rep == expected.rep


def test_peel_nested_addition() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")
    c = Expr.symbol("c")

    result = peel(lhs=(x + y) + z, rhs=c, target=SymbolTarget(x))
    expected = (c + (-z)) + (-y)
    assert result.rep == expected.rep


def test_peel_add_rejects_target_in_both_operands() -> None:
    x = Expr.symbol("x")
    c = Expr.symbol("c")

    with pytest.raises(IsolationError, match="Target occurs in both operands of addition"):
        peel(lhs=x + x, rhs=c, target=SymbolTarget(x))


def test_peel_add_rejects_target_not_found() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    z = Expr.symbol("z")
    c = Expr.symbol("c")

    with pytest.raises(IsolationError, match="Target not found in addition"):
        peel(lhs=y + z, rhs=c, target=SymbolTarget(x))


def test_peel_rejects_unsupported_expression_head() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")
    c = Expr.symbol("c")

    with pytest.raises(IsolationError, match="Cannot isolate through unsupported expression head"):
        peel(lhs=x * y, rhs=c, target=SymbolTarget(x))