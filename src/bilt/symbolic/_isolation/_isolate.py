__all__ = ["isolate"]


from bilt.symbolic._algebra._affine import (
    AffineAnalysisError,
    as_affine,
    subtract_affine,
)
from bilt.symbolic._algebra._expr_ops import (
    assert_binary_relation_tree,
    divide,
    is_zero,
    negate,
    simplify,
    zero,
)
from bilt.symbolic._atoms import SymbolType
from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import BiltTargetNotFoundError, BiltUnsupportedIsolationError
from bilt.symbolic._isolation._rules import isolate_by_inverse_rules
from bilt.symbolic._isolation._types import (
    IsolatedTree,
    IsolationPlan,
    OccurrenceProfile,
    PlannedTree,
    SimplifiedTree,
)
from bilt.symbolic._tree import Tree
from bilt.symbolic._walk import count_occurrences


def isolate(
    equation: Tree,
    target: Expr,
) -> IsolatedTree:
    """Return ``equation`` with ``target`` isolated on the left."""
    simplified = simplify_tree(equation)
    profile = occurrence_profile(simplified, target)

    if profile.target_missing:
        message = f"Target {target!r} does not occur in equation tree."
        raise BiltTargetNotFoundError(message)

    if profile.total_count == 1:
        isolated = _isolate_single_occurrence(
            simplified,
            target,
            profile,
        )
    else:
        isolated = _isolate_affine(
            simplified,
            target,
        )

    return IsolatedTree(
        relation_head=isolated.relation_head,
        target=isolated.target,
        rhs=simplify(isolated.rhs),
    )


def simplify_tree(tree: Tree) -> SimplifiedTree:
    assert_binary_relation_tree(tree)

    lhs = simplify(Expr(tree.args[0]))
    rhs = simplify(Expr(tree.args[1]))

    return SimplifiedTree(
        Tree(
            tree.head,
            (lhs.rep, rhs.rep),
        )
    )


def occurrence_profile(
    simplified: SimplifiedTree,
    target: Expr,
) -> OccurrenceProfile:
    tree = simplified.tree

    return OccurrenceProfile(
        lhs_count=count_occurrences(
            tree.args[0],
            target.rep,
        ),
        rhs_count=count_occurrences(
            tree.args[1],
            target.rep,
        ),
    )


def _isolate_single_occurrence(
    simplified: SimplifiedTree,
    target: Expr,
    profile: OccurrenceProfile,
) -> IsolatedTree:
    planned = PlannedTree(
        simplified_tree=simplified,
        target=target,
        profile=profile,
        plan=IsolationPlan(
            swap_sides=profile.rhs_count > 0,
        ),
    )

    return isolate_by_inverse_rules(planned)


def _isolate_affine(
    simplified: SimplifiedTree,
    target: Expr,
) -> IsolatedTree:
    tree = simplified.tree

    if not (target.rep.is_leaf and target.rep.head.atom_type is SymbolType):
        message = f"Affine isolation requires a symbolic target, got {target!r}."
        raise BiltUnsupportedIsolationError(message)

    symbol = str(target.rep.head.value)
    symbols = frozenset({symbol})

    try:
        left = as_affine(
            tree.args[0],
            symbols,
        )
        right = as_affine(
            tree.args[1],
            symbols,
        )
    except AffineAnalysisError as error:
        message = f"Target {target!r} occurs more than once and the equation is not affine in the target."
        raise BiltUnsupportedIsolationError(message) from error

    form = subtract_affine(left, right)

    coefficient = form.coefficients.get(
        symbol,
        zero(),
    )

    if is_zero(coefficient):
        message = f"Target {target!r} cancels from the equation."
        raise BiltUnsupportedIsolationError(message)

    rhs = divide(
        negate(form.constant),
        coefficient,
    )

    return IsolatedTree(
        relation_head=tree.head,
        target=target,
        rhs=Expr(rhs),
    )
