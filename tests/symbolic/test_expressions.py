import pytest

from bilt.symbolic._atoms import (
    ADD,
    EQUALS,
    INV,
    LESS_THAN,
    MUL,
    NEG,
    POW,
    SymbolType,
)
from bilt.symbolic._expressions import Expr, as_expr, as_tree, symbol_expr
from bilt.symbolic._tree import Tree, int_tree, symbol_tree


def test_symbol_factory_returns_expr_wrapping_symbol_tree() -> None:
    x = Expr.symbol("x")

    assert isinstance(x, Expr)
    assert x.rep is symbol_tree("x")
    assert x.tree is x.rep
    assert x.is_symbol is True
    assert x.is_number is False


def test_numeric_factories_return_exprs_wrapping_number_trees() -> None:
    integer_value = 1
    float_value = 1.5

    one = Expr.integer(integer_value)
    real = Expr.real(float_value)

    assert isinstance(one, Expr)
    assert isinstance(real, Expr)
    assert one.is_number is True
    assert real.is_number is True
    assert one.is_symbol is False
    assert real.is_symbol is False


def test_exprs_share_hash_consed_trees() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")
    y = Expr.symbol("y")

    assert x1 is not x2
    assert x1.rep is x2.rep
    assert x1.rep is not y.rep
    assert x1.structurally_equal(x2)
    assert not x1.structurally_equal(y)


def test_hash_uses_tree_identity() -> None:
    x1 = Expr.symbol("x")
    x2 = Expr.symbol("x")

    assert hash(x1) == hash(x2)


def test_as_tree_converts_supported_values() -> None:
    integer_value = 1
    float_value = 1.5

    x = Expr.symbol("x")

    assert as_tree(x) is x.rep
    assert as_tree(x.rep) is x.rep
    assert as_tree(integer_value) is int_tree(integer_value)
    assert as_tree(float_value).head.value == float_value


def test_as_tree_rejects_bool() -> None:
    value = True

    with pytest.raises(TypeError, match="Boolean values"):
        as_tree(value)


def test_as_expr_converts_supported_values() -> None:
    integer_value = 1

    x = Expr.symbol("x")
    one = as_expr(integer_value)

    assert as_expr(x) is x
    assert isinstance(one, Expr)
    assert one.rep is int_tree(integer_value)


def test_symbol_expr_constructs_symbol_expr() -> None:
    x = symbol_expr("x")

    assert isinstance(x, Expr)
    assert x.rep.head.atom_type is SymbolType
    assert x.rep.head.value == "x"


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
    exponent = 2

    x = Expr.symbol("x")

    expr = x**exponent

    assert expr.rep.head is POW
    assert expr.rep.args == (x.rep, int_tree(exponent))


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


def test_reverse_arithmetic_accepts_numbers() -> None:
    integer_value = 1

    x = Expr.symbol("x")

    add_expr = integer_value + x
    mul_expr = integer_value * x
    sub_expr = integer_value - x
    div_expr = integer_value / x

    assert add_expr.rep.head is ADD
    assert add_expr.rep.args == (int_tree(integer_value), x.rep)

    assert mul_expr.rep.head is MUL
    assert mul_expr.rep.args == (int_tree(integer_value), x.rep)

    assert sub_expr.rep.head is ADD
    assert sub_expr.rep.args[0] is int_tree(integer_value)
    assert sub_expr.rep.args[1].head is NEG

    assert div_expr.rep.head is MUL
    assert div_expr.rep.args[0] is int_tree(integer_value)
    assert div_expr.rep.args[1].head is INV


def test_eq_builds_relation_tree() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = x == y

    assert isinstance(relation, Tree)
    assert relation.head is EQUALS
    assert relation.args == (x.rep, y.rep)
    assert relation.is_relation is True
    assert relation.is_equality is True


def test_lt_builds_relation_tree() -> None:
    x = Expr.symbol("x")
    y = Expr.symbol("y")

    relation = x < y

    assert isinstance(relation, Tree)
    assert relation.head is LESS_THAN
    assert relation.args == (x.rep, y.rep)
    assert relation.is_relation is True
    assert relation.is_equality is False


def test_expr_cannot_be_used_as_bool() -> None:
    x = Expr.symbol("x")

    with pytest.raises(TypeError, match="Symbolic expressions cannot be used as booleans"):
        bool(x)
