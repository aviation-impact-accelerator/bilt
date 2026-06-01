import pytest

from bilt.symbolic._atoms import symbol
from bilt.symbolic._tree import Tree, float_tree, int_tree, symbol_tree


def test_symbol_tree_is_hash_consed() -> None:
    x1 = symbol_tree("x")
    x2 = symbol_tree("x")

    assert x1 is x2


def test_int_tree_is_hash_consed() -> None:
    n1 = int_tree(1)
    n2 = int_tree(1)

    assert n1 is n2


def test_float_tree_is_hash_consed() -> None:
    x1 = float_tree(1.5)
    x2 = float_tree(1.5)

    assert x1 is x2


def test_different_leaf_values_are_different_trees() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    assert x is not y


def test_different_leaf_atom_types_are_different_trees() -> None:
    symbol_one = symbol_tree("1")
    int_one = int_tree(1)
    float_one = float_tree(1.0)

    assert symbol_one is not int_one
    assert int_one is not float_one
    assert symbol_one is not float_one


def test_compound_tree_is_hash_consed() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    expr1 = Tree(symbol("add"), (x, y))
    expr2 = Tree(symbol("add"), (x, y))

    assert expr1 is expr2


def test_compound_tree_argument_order_matters() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    xy = Tree(symbol("add"), (x, y))
    yx = Tree(symbol("add"), (y, x))

    assert xy is not yx


def test_compound_tree_head_matters() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    add_expr = Tree(symbol("add"), (x, y))
    mul_expr = Tree(symbol("mul"), (x, y))

    assert add_expr is not mul_expr


def test_leaf_has_no_args() -> None:
    x = symbol_tree("x")

    assert x.head == symbol("x")
    assert x.args == ()
    assert x.is_leaf is True


def test_compound_tree_is_not_leaf() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")
    expr = Tree(symbol("add"), (x, y))

    assert expr.is_leaf is False


def test_equality_is_identity_based() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    assert x != y
    assert x != object()


def test_hash_is_identity_based() -> None:
    x = symbol_tree("x")

    assert hash(x) == id(x)


def test_head_cannot_be_reassigned() -> None:
    x = symbol_tree("x")

    with pytest.raises(AttributeError, match="immutable"):
        x.head = symbol("y")


def test_args_cannot_be_reassigned() -> None:
    x = symbol_tree("x")

    with pytest.raises(AttributeError, match="immutable"):
        x.args = (symbol_tree("y"),)


def test_head_cannot_be_deleted() -> None:
    x = symbol_tree("x")

    with pytest.raises(AttributeError, match="immutable"):
        del x.head


def test_args_cannot_be_deleted() -> None:
    x = symbol_tree("x")

    with pytest.raises(AttributeError, match="immutable"):
        del x.args


def test_tree_cannot_gain_extra_attributes() -> None:
    x = symbol_tree("x")

    with pytest.raises(AttributeError):
        x.extra = 1  # type: ignore[attr-defined]


def test_str_for_leaf() -> None:
    assert str(symbol_tree("x")) == "x"
    assert str(int_tree(1)) == "1"
    assert str(float_tree(1.5)) == "1.5"


def test_str_for_compound_tree() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    expr = Tree(symbol("add"), (x, y))

    assert str(expr) == "add(x, y)"
