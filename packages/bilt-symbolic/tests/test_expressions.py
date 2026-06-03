from bilt.symbolic._expressions import Expr


def test_symbol_factory_returns_expr() -> None:
    x = Expr.symbol("x")

    assert isinstance(x, Expr)


def test_symbol_factory_uses_hash_consed_tree() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")

    assert x1 is not x2
    assert x1.rep is x2.rep


def test_structural_equality_uses_tree_identity() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")
    y = Expr.symbol("y")

    assert x1 == x2
    assert x1 != y


def test_hash_uses_tree_identity() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")

    assert hash(x1) == hash(x2)


def test_is_symbol() -> None:
    x = Expr.symbol("x")
    n = Expr.integer(1)

    assert x.is_symbol
    assert not n.is_symbol


def test_is_number_for_integer() -> None:
    n = Expr.integer(1)

    assert n.is_number


def test_is_number_for_real() -> None:
    x = Expr.real(1.5)

    assert x.is_number


def test_symbol_is_not_number() -> None:
    x = Expr.symbol("x")

    assert not x.is_number
