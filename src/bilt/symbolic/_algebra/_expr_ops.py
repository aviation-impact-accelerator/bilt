__all__ = [
    "BINARY_ARG_COUNT",
    "UNARY_ARG_COUNT",
    "add",
    "assert_binary_relation_tree",
    "count_tree_occurrences",
    "divide",
    "expr_args",
    "expr_contains",
    "is_minus_one",
    "is_one",
    "is_zero",
    "make_add",
    "make_div",
    "make_inv",
    "make_mul",
    "make_neg",
    "make_sub",
    "multiply",
    "multiply_many",
    "negate",
    "one",
    "relation_lhs",
    "relation_rhs",
    "same_expr",
    "same_tree",
    "simplify",
    "simplify_add",
    "simplify_inv",
    "simplify_mul",
    "simplify_neg",
    "split_exprs_by_target",
    "subtract",
    "tree_lhs_expr",
    "tree_rhs_expr",
    "unary_child",
    "zero",
]
import typing

from bilt.symbolic._atoms import (
    ADD,
    INV,
    MUL,
    NEG,
    FloatType,
    IntType,
)
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import UnsupportedIsolationError
from bilt.symbolic._tree import Tree, int_tree
from bilt.symbolic._walk import contains, count_occurrences

if typing.TYPE_CHECKING:
    from collections.abc import Iterable


BINARY_ARG_COUNT = 2
UNARY_ARG_COUNT = 1


def simplify(expr: Expr) -> Expr:
    head = expr.rep.head

    match head:
        case _ if head is ADD:
            return simplify_add(expr)
        case _ if head is MUL:
            return simplify_mul(expr)
        case _ if head is NEG:
            return simplify_neg(expr)
        case _ if head is INV:
            return simplify_inv(expr)
        case _:
            return expr


def simplify_add(expr: Expr) -> Expr:
    terms = tuple(simplify(Expr(child)) for child in expr.rep.args)

    flattened: list[Expr] = []
    for term in terms:
        if term.rep.head is ADD:
            flattened.extend(Expr(child) for child in term.rep.args)
        else:
            flattened.append(term)

    if len(flattened) == 1:
        return flattened[0]

    return make_add(*flattened)


def simplify_mul(expr: Expr) -> Expr:
    factors = tuple(simplify(Expr(child)) for child in expr.rep.args)

    flattened: list[Expr] = []
    for factor in factors:
        if factor.rep.head is MUL:
            flattened.extend(Expr(child) for child in factor.rep.args)
        else:
            flattened.append(factor)

    if len(flattened) == 1:
        return flattened[0]

    return make_mul(*flattened)


def simplify_neg(expr: Expr) -> Expr:
    child = simplify(unary_child(expr))

    if child.rep.head is NEG:
        return unary_child(child)

    return make_neg(child)


def simplify_inv(expr: Expr) -> Expr:
    child = simplify(unary_child(expr))

    if child.rep.head is INV:
        return unary_child(child)

    return make_inv(child)


def add(
    left: Tree,
    right: Tree,
) -> Tree:
    """Add two expression trees and simplify trivial identities."""
    if is_zero(left):
        return right

    if is_zero(right):
        return left

    return simplify(
        make_add(
            Expr(left),
            Expr(right),
        )
    ).rep


def subtract(
    left: Tree,
    right: Tree,
) -> Tree:
    """Subtract ``right`` from ``left``."""
    if same_tree(left, right):
        return zero()

    if is_zero(right):
        return left

    if is_zero(left):
        return negate(right)

    return add(
        left,
        negate(right),
    )


def multiply(
    left: Tree,
    right: Tree,
) -> Tree:
    """Multiply two expression trees and simplify trivial identities."""
    if is_zero(left) or is_zero(right):
        return zero()

    if is_one(left):
        return right

    if is_one(right):
        return left

    if is_minus_one(left):
        return negate(right)

    if is_minus_one(right):
        return negate(left)

    return simplify(
        make_mul(
            Expr(left),
            Expr(right),
        )
    ).rep


def multiply_many(
    trees: Iterable[Tree],
) -> Tree:
    """Multiply a sequence of expression trees."""
    result = one()

    for tree in trees:
        result = multiply(result, tree)

    return result


def negate(tree: Tree) -> Tree:
    """Negate an expression tree."""
    if is_zero(tree):
        return tree

    return simplify(make_neg(Expr(tree))).rep


def divide(
    numerator: Tree,
    denominator: Tree,
) -> Tree:
    """Divide one expression tree by another."""
    if is_zero(numerator):
        return zero()

    if is_one(denominator):
        return numerator

    if is_minus_one(denominator):
        return negate(numerator)

    return simplify(
        make_div(
            Expr(numerator),
            Expr(denominator),
        )
    ).rep


def zero() -> Tree:
    return int_tree(0)


def one() -> Tree:
    return int_tree(1)


def is_zero(tree: Tree) -> bool:
    return tree.is_leaf and tree.head.atom_type in (IntType, FloatType) and tree.head.value == 0


def is_one(tree: Tree) -> bool:
    return tree.is_leaf and tree.head.atom_type in (IntType, FloatType) and tree.head.value == 1


def is_minus_one(tree: Tree) -> bool:
    return tree.head is NEG and len(tree.args) == UNARY_ARG_COUNT and is_one(tree.args[0])


def split_exprs_by_target(
    exprs: tuple[Expr, ...],
    target: Expr,
) -> tuple[tuple[Expr, ...], tuple[Expr, ...]]:
    target_exprs: list[Expr] = []
    other_exprs: list[Expr] = []

    for expr in exprs:
        if contains(expr.rep, target.rep):
            target_exprs.append(expr)
        else:
            other_exprs.append(expr)

    return tuple(target_exprs), tuple(other_exprs)


def make_add(*terms: Expr) -> Expr:
    if not terms:
        message = "Cannot build an empty addition."
        raise ValueError(message)

    return Expr(Tree(ADD, tuple(term.rep for term in terms)))


def make_mul(*factors: Expr) -> Expr:
    if not factors:
        message = "Cannot build an empty multiplication."
        raise ValueError(message)

    return Expr(Tree(MUL, tuple(factor.rep for factor in factors)))


def make_neg(expr: Expr) -> Expr:
    return Expr(Tree(NEG, (expr.rep,)))


def make_inv(expr: Expr) -> Expr:
    return Expr(Tree(INV, (expr.rep,)))


def make_sub(lhs: Expr, rhs: Expr) -> Expr:
    return make_add(lhs, make_neg(rhs))


def make_div(lhs: Expr, rhs: Expr) -> Expr:
    return make_mul(lhs, make_inv(rhs))


def assert_binary_relation_tree(tree: Tree) -> None:
    if len(tree.args) != BINARY_ARG_COUNT:
        message = f"Expected a binary equation/relation tree with {BINARY_ARG_COUNT} args, got {len(tree.args)}."
        raise UnsupportedIsolationError(message)


def relation_lhs(tree: Tree) -> Tree:
    assert_binary_relation_tree(tree)
    return tree.args[0]


def relation_rhs(tree: Tree) -> Tree:
    assert_binary_relation_tree(tree)
    return tree.args[1]


def tree_lhs_expr(tree: Tree) -> Expr:
    return Expr(relation_lhs(tree))


def tree_rhs_expr(tree: Tree) -> Expr:
    return Expr(relation_rhs(tree))


def expr_args(expr: Expr) -> tuple[Expr, ...]:
    return tuple(Expr(child) for child in expr.rep.args)


def unary_child(expr: Expr) -> Expr:
    if len(expr.rep.args) != UNARY_ARG_COUNT:
        message = f"Expected unary expression {expr!r} to have {UNARY_ARG_COUNT} arg, got {len(expr.rep.args)}."
        raise UnsupportedIsolationError(message)

    return Expr(expr.rep.args[0])


def expr_contains(expr: Expr, target: Expr) -> bool:
    return contains(expr.rep, target.rep)


def count_tree_occurrences(tree: Tree, target: Tree) -> int:
    return count_occurrences(tree, target)


def same_expr(left: Expr, right: Expr) -> bool:
    """Internal structural equality.

    Do not use Expr.__eq__, because Expr.__eq__ constructs a relation tree.
    """
    return same_tree(left.rep, right.rep)


def same_tree(left: Tree, right: Tree) -> bool:
    if left is right:
        return True

    if left.head != right.head:
        return False

    if len(left.args) != len(right.args):
        return False

    return all(
        same_tree(left_child, right_child)
        for left_child, right_child in zip(
            left.args,
            right.args,
            strict=True,
        )
    )
