from bilt.symbolic._atoms import symbol
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.symbolic._walk import contains, walk_postorder, walk_preorder


def test_walk_preorder_leaf() -> None:
    x = symbol_tree("x")

    assert walk_preorder(x) == [x]


def test_walk_postorder_leaf() -> None:
    x = symbol_tree("x")

    assert walk_postorder(x) == [x]


def test_contains_leaf_self() -> None:
    x = symbol_tree("x")

    assert contains(x, x)


def test_contains_leaf_other() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    assert not contains(x, y)


def test_walk_preorder_simple_tree() -> None:
    add = symbol("+")
    x = symbol_tree("x")
    y = symbol_tree("y")
    expr = Tree(add, (x, y))

    assert walk_preorder(expr) == [expr, x, y]


def test_walk_postorder_simple_tree() -> None:
    add = symbol("+")
    x = symbol_tree("x")
    y = symbol_tree("y")
    expr = Tree(add, (x, y))

    assert walk_postorder(expr) == [x, y, expr]


def test_contains_direct_child() -> None:
    add = symbol("+")
    x = symbol_tree("x")
    y = symbol_tree("y")
    expr = Tree(add, (x, y))

    assert contains(expr, x)
    assert contains(expr, y)


def test_contains_missing_child() -> None:
    add = symbol("+")
    x = symbol_tree("x")
    y = symbol_tree("y")
    z = symbol_tree("z")
    expr = Tree(add, (x, y))

    assert not contains(expr, z)


def test_walk_preorder_nested_tree() -> None:
    add = symbol("+")
    mul = symbol("*")

    x = symbol_tree("x")
    y = symbol_tree("y")
    z = symbol_tree("z")

    product = Tree(mul, (y, z))
    expr = Tree(add, (x, product))

    assert walk_preorder(expr) == [expr, x, product, y, z]


def test_walk_postorder_nested_tree() -> None:
    add = symbol("+")
    mul = symbol("*")

    x = symbol_tree("x")
    y = symbol_tree("y")
    z = symbol_tree("z")

    product = Tree(mul, (y, z))
    expr = Tree(add, (x, product))

    assert walk_postorder(expr) == [x, y, z, product, expr]


def test_contains_nested_child() -> None:
    add = symbol("+")
    mul = symbol("*")

    x = symbol_tree("x")
    y = symbol_tree("y")
    z = symbol_tree("z")

    product = Tree(mul, (y, z))
    expr = Tree(add, (x, product))

    assert contains(expr, product)
    assert contains(expr, y)
    assert contains(expr, z)


def test_contains_shared_subtree() -> None:
    add = symbol("+")
    neg = symbol("neg")

    x = symbol_tree("x")
    shared = Tree(neg, (x,))
    expr = Tree(add, (shared, shared))

    assert expr.args[0] is expr.args[1]
    assert contains(expr, shared)
    assert contains(expr, x)


def test_walk_preorder_visits_shared_subtree_occurrences() -> None:
    add = symbol("+")
    neg = symbol("neg")

    x = symbol_tree("x")
    shared = Tree(neg, (x,))
    expr = Tree(add, (shared, shared))

    assert walk_preorder(expr) == [
        expr,
        shared,
        x,
        shared,
        x,
    ]


def test_walk_postorder_visits_shared_subtree_occurrences() -> None:
    add = symbol("+")
    neg = symbol("neg")

    x = symbol_tree("x")
    shared = Tree(neg, (x,))
    expr = Tree(add, (shared, shared))

    assert walk_postorder(expr) == [
        x,
        shared,
        x,
        shared,
        expr,
    ]


def test_contains_hash_consed_equivalent_leaf() -> None:
    add = symbol("+")

    x1 = symbol_tree("x")
    x2 = symbol_tree("x")
    y = symbol_tree("y")

    expr = Tree(add, (x1, y))

    assert x1 is x2
    assert contains(expr, x2)


def test_contains_hash_consed_equivalent_subtree() -> None:
    add = symbol("+")
    neg = symbol("neg")

    x = symbol_tree("x")
    y = symbol_tree("y")

    neg_x_1 = Tree(neg, (x,))
    neg_x_2 = Tree(neg, (x,))

    expr = Tree(add, (neg_x_1, y))

    assert neg_x_1 is neg_x_2
    assert contains(expr, neg_x_2)


def test_contains_similar_but_different_subtree() -> None:
    add = symbol("+")
    neg = symbol("neg")

    x = symbol_tree("x")
    y = symbol_tree("y")

    neg_x = Tree(neg, (x,))
    neg_y = Tree(neg, (y,))

    expr = Tree(add, (neg_x,))

    assert not contains(expr, neg_y)
