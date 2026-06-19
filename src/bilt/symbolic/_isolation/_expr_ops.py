__all__ = [
    "BINARY_ARG_COUNT",
    "UNARY_ARG_COUNT",
    "assert_binary_relation_tree",
    "count_tree_occurrences",
    "expr_args",
    "expr_contains",
    "make_add",
    "make_div",
    "make_inv",
    "make_mul",
    "make_neg",
    "make_relation_tree",
    "make_sub",
    "same_expr",
    "same_tree",
    "simplify",
    "simplify_add",
    "simplify_inv",
    "simplify_mul",
    "simplify_neg",
    "split_exprs_by_target",
    "tree_hash",
    "tree_lhs_expr",
    "tree_lhs_tree",
    "tree_rhs_expr",
    "tree_rhs_tree",
    "unary_child",
]

from bilt.symbolic._atoms import ADD, INV, MUL, NEG, Atom
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import UnsupportedIsolationError
from bilt.symbolic._tree import Tree
from bilt.symbolic._walk import contains, count_occurrences

BINARY_ARG_COUNT = 2
UNARY_ARG_COUNT = 1


def simplify(expr: Expr) -> Expr:
    head = expr.rep.head

    if head is ADD:
        return simplify_add(expr)

    if head is MUL:
        return simplify_mul(expr)

    if head is NEG:
        return simplify_neg(expr)

    if head is INV:
        return simplify_inv(expr)

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


def make_relation_tree(relation_head: Atom, lhs: Expr, rhs: Expr) -> Tree:
    return Tree(relation_head, (lhs.rep, rhs.rep))


def make_add(*terms: Expr) -> Expr:
    if not terms:
        message = "Cannot build an empty addition."
        raise ValueError(message)

    if len(terms) == 1:
        return terms[0]

    return Expr(Tree(ADD, tuple(term.rep for term in terms)))


def make_mul(*factors: Expr) -> Expr:
    if not factors:
        message = "Cannot build an empty multiplication."
        raise ValueError(message)

    if len(factors) == 1:
        return factors[0]

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


def tree_lhs_tree(tree: Tree) -> Tree:
    assert_binary_relation_tree(tree)
    return tree.args[0]


def tree_rhs_tree(tree: Tree) -> Tree:
    assert_binary_relation_tree(tree)
    return tree.args[1]


def tree_lhs_expr(tree: Tree) -> Expr:
    return Expr(tree_lhs_tree(tree))


def tree_rhs_expr(tree: Tree) -> Expr:
    return Expr(tree_rhs_tree(tree))


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


def tree_hash(tree: Tree) -> int:
    return hash((tree.head, tuple(tree_hash(child) for child in tree.args)))


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
        same_tree(left_child, right_child) for left_child, right_child in zip(left.args, right.args, strict=True)
    )
