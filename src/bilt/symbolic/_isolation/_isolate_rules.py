__all__ = [
    "isolate_add_expr",
    "isolate_by_inverse_rules",
    "isolate_expr_by_inverse_rules",
    "isolate_mul_expr",
]

import typing

from bilt.symbolic._isolation._errors import UnsupportedIsolationError
from bilt.symbolic._isolation._expr_ops import (
    as_add_terms,
    as_mul_factors,
    is_add,
    is_inv,
    is_mul,
    is_neg,
    make_add,
    make_div,
    make_inv,
    make_mul,
    make_neg,
    make_sub,
    same_expr,
    simplify,
    split_exprs_by_target,
    tree_lhs_expr,
    tree_rhs_expr,
    unary_child,
)
from bilt.symbolic._isolation._isolate_types import (
    IsolatedTree,
    IsolationStrategy,
    PlannedTree,
)

if typing.TYPE_CHECKING:
    from bilt.symbolic._atoms import Atom
    from bilt.symbolic._expressions import Expr


def isolate_by_inverse_rules(planned: PlannedTree) -> IsolatedTree:
    tree = planned.simplified_tree.tree
    relation_head = tree.head
    lhs = tree_lhs_expr(tree)
    rhs = tree_rhs_expr(tree)
    target = planned.target
    strategy = planned.plan.strategy

    if strategy is IsolationStrategy.ALREADY_ISOLATED:
        return IsolatedTree(relation_head=relation_head, target=target, rhs=rhs)

    if strategy is IsolationStrategy.SWAP_SIDES:
        return IsolatedTree(relation_head=relation_head, target=target, rhs=lhs)

    if strategy is IsolationStrategy.SWAP_THEN_INVERSE_RULES:
        lhs, rhs = rhs, lhs

    return isolate_expr_by_inverse_rules(
        lhs=lhs,
        rhs=rhs,
        target=target,
        relation_head=relation_head,
    )


# =============================================================================
# Inverse-rule isolation
# =============================================================================


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

    if is_neg(lhs):
        child = unary_child(lhs)
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_neg(rhs),
            target=target,
            relation_head=relation_head,
        )

    if is_inv(lhs):
        child = unary_child(lhs)
        return isolate_expr_by_inverse_rules(
            lhs=child,
            rhs=make_inv(rhs),
            target=target,
            relation_head=relation_head,
        )

    if is_add(lhs):
        new_lhs, new_rhs = isolate_add_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(
            lhs=new_lhs,
            rhs=new_rhs,
            target=target,
            relation_head=relation_head,
        )

    if is_mul(lhs):
        new_lhs, new_rhs = isolate_mul_expr(lhs, rhs, target)
        return isolate_expr_by_inverse_rules(
            lhs=new_lhs,
            rhs=new_rhs,
            target=target,
            relation_head=relation_head,
        )

    message = f"Cannot isolate {target!r} from {lhs!r}."
    raise UnsupportedIsolationError(message)


def isolate_add_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    terms = as_add_terms(lhs)
    target_terms, other_terms = split_exprs_by_target(terms, target)

    if len(target_terms) != 1:
        message = f"Expected one target-containing additive term, got {len(target_terms)}."
        raise UnsupportedIsolationError(message)

    target_part = target_terms[0]

    if not other_terms:
        return target_part, rhs

    other_part = make_add(*other_terms)
    new_rhs = make_sub(rhs, other_part)

    return simplify(target_part), simplify(new_rhs)


def isolate_mul_expr(
    lhs: Expr,
    rhs: Expr,
    target: Expr,
) -> tuple[Expr, Expr]:
    factors = as_mul_factors(lhs)
    target_factors, other_factors = split_exprs_by_target(factors, target)

    if len(target_factors) != 1:
        message = f"Expected one target-containing multiplicative factor, got {len(target_factors)}."
        raise UnsupportedIsolationError(message)

    target_part = target_factors[0]

    if not other_factors:
        return target_part, rhs

    other_part = make_mul(*other_factors)
    new_rhs = make_div(rhs, other_part)

    return simplify(target_part), simplify(new_rhs)
