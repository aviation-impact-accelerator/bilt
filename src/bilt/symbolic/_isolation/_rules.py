__all__ = [
    "isolate_add_expr",
    "isolate_by_inverse_rules",
    "isolate_expr_by_inverse_rules",
    "isolate_mul_expr",
]

from typing import TYPE_CHECKING

from bilt.symbolic._atoms import ADD, INV, MUL, NEG
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import BiltUnsupportedIsolationError
from bilt.symbolic._isolation._expr_ops import (
    make_add,
    make_div,
    make_inv,
    make_mul,
    make_neg,
    make_sub,
    same_expr,
    simplify,
    split_exprs_by_target,
    unary_child,
)
from bilt.symbolic._isolation._types import IsolatedTree, PlannedTree

if TYPE_CHECKING:
    from bilt.symbolic._atoms import Atom


def isolate_by_inverse_rules(planned: PlannedTree) -> IsolatedTree:
    tree = planned.simplified_tree.tree
    relation_head = tree.head
    lhs = Expr(tree.args[0])
    rhs = Expr(tree.args[1])
    target = planned.target

    if planned.plan.swap_sides:
        lhs, rhs = rhs, lhs

    return isolate_expr_by_inverse_rules(
        lhs=lhs,
        rhs=rhs,
        target=target,
        relation_head=relation_head,
    )


def isolate_expr_by_inverse_rules(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
    relation_head: Atom,
) -> IsolatedTree:
    lhs = simplify(lhs)
    rhs = simplify(rhs)

    if same_expr(lhs, target):
        return IsolatedTree(relation_head=relation_head, target=target, rhs=rhs)

    head = lhs.rep.head

    if head is NEG:
        child = unary_child(lhs)
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_neg(rhs),
            target=target,
            relation_head=relation_head,
        )

    if head is INV:
        child = unary_child(lhs)
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_inv(rhs),
            target=target,
            relation_head=relation_head,
        )

    if head is ADD:
        new_lhs, new_rhs = isolate_add_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(
            lhs=new_lhs,
            rhs=new_rhs,
            target=target,
            relation_head=relation_head,
        )

    if head is MUL:
        new_lhs, new_rhs = isolate_mul_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(
            lhs=new_lhs,
            rhs=new_rhs,
            target=target,
            relation_head=relation_head,
        )

    message = f"Cannot isolate {target!r} from {lhs!r}."
    raise BiltUnsupportedIsolationError(message)


def isolate_add_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    terms = tuple(Expr(child) for child in lhs.rep.args)
    target_terms, other_terms = split_exprs_by_target(terms, target)

    if len(target_terms) != 1:
        message = f"Expected one target-containing additive term, got {len(target_terms)}."
        raise BiltUnsupportedIsolationError(message)

    target_part = target_terms[0]

    if not other_terms:
        return simplify(target_part), rhs

    other_part = make_add(*other_terms)
    new_rhs = make_sub(rhs, other_part)

    return simplify(target_part), simplify(new_rhs)


def isolate_mul_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    factors = tuple(Expr(child) for child in lhs.rep.args)
    target_factors, other_factors = split_exprs_by_target(factors, target)

    if len(target_factors) != 1:
        message = f"Expected one target-containing multiplicative factor, got {len(target_factors)}."
        raise BiltUnsupportedIsolationError(message)

    target_part = target_factors[0]

    if not other_factors:
        return simplify(target_part), rhs

    other_part = make_mul(*other_factors)
    new_rhs = make_div(rhs, other_part)

    return simplify(target_part), simplify(new_rhs)
