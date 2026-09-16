import pytest

from bilt.symbolic._atoms import (
    ADD,
    EQUALS,
    LESS_THAN,
    MUL,
    Atom,
    AxisType,
    axis,
    symbol,
)
from bilt.symbolic._errors import TreeMutationError
from bilt.symbolic._tree import (
    Tree,
    axis_tree,
    equality_tree,
    float_tree,
    int_tree,
    relation_tree,
    symbol_tree,
)


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


def test_axis_tree_is_hash_consed() -> None:
    axis1 = axis_tree(1)
    axis2 = axis_tree(1)

    assert axis1 is axis2


def test_axis_tree_has_axis_head() -> None:
    tree = axis_tree(1)

    assert tree.head is axis(1)
    assert tree.head.atom_type is AxisType
    assert tree.head.value == 1
    assert tree.args == ()
    assert tree.is_leaf is True


def test_axis_tree_is_distinct_from_int_tree() -> None:
    assert axis_tree(1) is not int_tree(1)


def test_different_leaf_values_are_different_trees() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    assert x is not y


def test_leaf_tree_has_head_and_no_args() -> None:
    x = symbol_tree("x")

    assert x.head is symbol("x")
    assert x.args == ()
    assert x.is_leaf is True


def test_compound_tree_is_hash_consed() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    expr1 = Tree(ADD, (x, y))
    expr2 = Tree(ADD, (x, y))

    assert expr1 is expr2


def test_compound_tree_preserves_head_and_args() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    expr = Tree(ADD, (x, y))

    assert expr.head is ADD
    assert expr.args == (x, y)
    assert expr.is_leaf is False


def test_compound_tree_argument_order_matters() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    xy = Tree(ADD, (x, y))
    yx = Tree(ADD, (y, x))

    assert xy is not yx


def test_compound_tree_head_matters() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    add_expr = Tree(ADD, (x, y))
    mul_expr = Tree(MUL, (x, y))

    assert add_expr is not mul_expr


def test_tree_equality_is_identity_based() -> None:
    x = symbol_tree("x")
    x_again = symbol_tree("x")
    y = symbol_tree("y")

    assert x is x_again
    assert x == x_again
    assert x != y
    assert x != object()


def test_tree_hash_is_identity_based() -> None:
    x = symbol_tree("x")

    assert hash(x) == id(x)


def test_tree_is_immutable() -> None:
    x = symbol_tree("x")

    with pytest.raises(TreeMutationError, match="immutable"):
        x.head = symbol("y")

    with pytest.raises(TreeMutationError, match="immutable"):
        x.args = ()

    with pytest.raises(TreeMutationError, match="immutable"):
        del x.head

    with pytest.raises(TreeMutationError, match="immutable"):
        x.extra = 1  # type: ignore[attr-defined]


def test_str_for_leaf_tree() -> None:
    assert str(symbol_tree("x")) == "x"
    assert str(int_tree(1)) == "1"
    assert str(float_tree(1.5)) == "1.5"


def test_str_for_compound_tree() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    expr = Tree(ADD, (x, y))

    assert str(expr) == "add(x, y)"


@pytest.mark.parametrize(
    ("operator", "operator_text"),
    (
        (EQUALS, "=="),
        (LESS_THAN, "<"),
    ),
)
def test_relation_tree_constructs_from_relation_operator_atom(
    operator: Atom,
    operator_text: str,
) -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    relation = relation_tree(operator, x, y)

    assert relation.head is operator
    assert relation.args == (x, y)
    assert relation.is_relation is True
    assert relation.operator is operator
    assert relation.left is x
    assert relation.right is y
    assert str(relation) == f"(x {operator_text} y)"


@pytest.mark.parametrize(
    "operator",
    (
        "==",
        "<",
    ),
)
def test_relation_tree_constructs_from_operator_string(operator: str) -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    relation = relation_tree(operator, x, y)

    assert relation.args == (x, y)
    assert relation.is_relation is True
    assert str(relation) == f"(x {operator} y)"


def test_equality_tree_constructs_equality_relation() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    relation = equality_tree(x, y)

    assert relation.head is EQUALS
    assert relation.args == (x, y)
    assert relation.is_relation is True
    assert relation.is_equality is True
    assert relation.left is x
    assert relation.right is y


def test_non_relation_tree_rejects_relation_accessors() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    expr = Tree(ADD, (x, y))

    assert expr.is_relation is False
    assert expr.is_equality is False

    with pytest.raises(ValueError, match="not a relation"):
        _ = expr.operator

    with pytest.raises(ValueError, match="no left-hand side"):
        _ = expr.left

    with pytest.raises(ValueError, match="no right-hand side"):
        _ = expr.right


def test_relation_tree_rejects_non_relation_operator_atom() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    with pytest.raises(TypeError, match="Expected relation operator atom"):
        relation_tree(ADD, x, y)


def test_relation_tree_rejects_unknown_operator_string() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    with pytest.raises(ValueError, match="Unknown relation operator"):
        relation_tree("===", x, y)
