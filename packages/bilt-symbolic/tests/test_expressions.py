from bilt.symbolic._equalities import Equality
from bilt.symbolic._expressions import ADD, INV, MUL, NEG, POW, Expr


def test_symbol_factory_returns_expr() -> None:
    x = Expr.symbol("x")

    assert isinstance(x, Expr)


def test_symbol_factory_uses_hash_consed_tree() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")

    assert x1 is not x2
    assert x1.rep is x2.rep


def test_expr_eq_builds_equality() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    equality = x == y

    assert isinstance(equality, Equality)
    assert equality.lhs is x
    assert equality.rhs is y


def test_expr_structural_identity_uses_hash_consed_tree() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")
    y = Expr.symbol("y")

    assert x1.rep is x2.rep
    assert x1.rep is not y.rep


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


def test_add_builds_add_tree() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    expr = x + y

    assert expr.rep.head is ADD
    assert expr.rep.args == (x.rep, y.rep)


def test_mul_builds_mul_tree() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    expr = x * y

    assert expr.rep.head is MUL
    assert expr.rep.args == (x.rep, y.rep)


def test_pow_builds_pow_tree() -> None:
    x = Expr.symbol("x")
    two = Expr.integer(2)

    expr = x**two

    assert expr.rep.head is POW
    assert expr.rep.args == (x.rep, two.rep)


def test_neg_builds_neg_tree() -> None:
    x = Expr.symbol("x")

    expr = -x

    assert expr.rep.head is NEG
    assert expr.rep.args == (x.rep,)


def test_sub_builds_add_of_neg() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    expr = x - y

    assert expr.rep.head is ADD
    assert expr.rep.args[0] is x.rep

    neg_y = expr.rep.args[1]
    assert neg_y.head is NEG
    assert neg_y.args == (y.rep,)


def test_truediv_builds_mul_of_inv() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    expr = x / y

    assert expr.rep.head is MUL
    assert expr.rep.args[0] is x.rep

    inv_y = expr.rep.args[1]
    assert inv_y.head is INV
    assert inv_y.args == (y.rep,)
