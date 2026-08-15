from bilt.symbolic._atoms import symbol
from bilt.symbolic._tree import Tree, symbol_tree
from bilt.symbolic._walk import (
    contains,
    count_occurrences,
    subtrees,
    symbol_names,
)


def test_contains_nested_subtree() -> None:
    add = symbol("+")
    mul = symbol("*")

    x = symbol_tree("x")
    y = symbol_tree("y")
    z = symbol_tree("z")

    product = Tree(mul, (y, z))
    expr = Tree(add, (x, product))

    assert contains(expr, product)
    assert contains(expr, y)
    assert not contains(expr, symbol_tree("missing"))


def test_contains_hash_consed_equivalent_subtree() -> None:
    neg = symbol("neg")
    x = symbol_tree("x")

    first = Tree(neg, (x,))
    second = Tree(neg, (x,))

    assert first is second
    assert contains(first, second)


def test_subtrees_returns_each_distinct_subtree_once() -> None:
    add = symbol("+")
    neg = symbol("neg")

    x = symbol_tree("x")
    shared = Tree(neg, (x,))
    expr = Tree(add, (shared, shared))

    assert subtrees(expr) == {expr, shared, x}


def test_count_occurrences_counts_shared_references() -> None:
    add = symbol("+")
    x = symbol_tree("x")
    expr = Tree(add, (x, x))

    assert count_occurrences(expr, x) == 2


def test_count_occurrences_returns_zero_when_missing() -> None:
    x = symbol_tree("x")
    y = symbol_tree("y")

    assert count_occurrences(x, y) == 0


def test_symbol_names_returns_distinct_symbols() -> None:
    add = symbol("+")
    mul = symbol("*")

    x = symbol_tree("x")
    y = symbol_tree("y")

    expr = Tree(
        add,
        (
            x,
            Tree(mul, (x, y)),
        ),
    )

    assert symbol_names(expr) == {"x", "y"}
