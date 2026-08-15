__all__ = ["equation_to_ast", "isolate"]

import typing

from bilt.symbolic._expressions import Expr
from bilt.symbolic._isolation._errors import (
    TargetNotFoundError,
    TargetOnBothSidesError,
    UnsupportedIsolationError,
)
from bilt.symbolic._algebra._expr_ops import (
    assert_binary_relation_tree,
    simplify,
)
from bilt.symbolic._isolation._rules import isolate_by_inverse_rules
from bilt.symbolic._isolation._to_ast import isolated_tree_to_functiondef
from bilt.symbolic._isolation._types import (
    AnalysedTree,
    IsolatedTree,
    IsolationPlan,
    OccurrenceProfile,
    PlannedTree,
    SimplifiedTree,
)
from bilt.symbolic._tree import Tree
from bilt.symbolic._walk import count_occurrences

if typing.TYPE_CHECKING:
    import ast

    from bilt.core._equation import BasicEquation


def equation_to_ast(
    equation: BasicEquation,
    target: str,
) -> ast.FunctionDef:
    if target not in equation.__bilt_equation_symbols__:
        message = f"Target {target!r} is not a parameter of the equation {equation!s}."
        raise TargetNotFoundError(message)
    target_expr = Expr.symbol(target)
    isolated = isolate(equation.__bilt_equation_tree__, target_expr)
    return isolated_tree_to_functiondef(
        isolated,
        parameter_order=equation.__bilt_parameters__,
    )


def isolate(equation: Tree, target: Expr) -> IsolatedTree:
    """Return a relation tree with ``target`` isolated on the left-hand side.

    ``equation`` is expected to be a binary relation tree whose two children are
    expression trees. The returned tree keeps the same relation head as the input
    and has ``target.rep`` as its first child.
    """
    simplified = simplify_tree(equation)
    analysed = analyse_occurrences(simplified, target)
    planned = choose_isolation_plan(analysed)
    isolated = isolate_by_inverse_rules(planned)

    return IsolatedTree(
        relation_head=isolated.relation_head,
        target=isolated.target,
        rhs=simplify(isolated.rhs),
    )


def simplify_tree(tree: Tree) -> SimplifiedTree:
    assert_binary_relation_tree(tree)

    lhs = simplify(Expr(tree.args[0]))
    rhs = simplify(Expr(tree.args[1]))

    return SimplifiedTree(Tree(tree.head, (lhs.rep, rhs.rep)))


def analyse_occurrences(
    simplified: SimplifiedTree,
    target: Expr,
) -> AnalysedTree:
    tree = simplified.tree

    lhs_count = count_occurrences(tree.args[0], target.rep)
    rhs_count = count_occurrences(tree.args[1], target.rep)

    profile = OccurrenceProfile(
        lhs_count=lhs_count,
        rhs_count=rhs_count,
    )

    if profile.target_missing:
        message = f"Target {target!r} does not occur in equation tree."
        raise TargetNotFoundError(message)

    if profile.target_on_both_sides:
        message = f"Target {target!r} occurs on both sides."
        raise TargetOnBothSidesError(message)

    if profile.total_count > 1:
        message = f"Target {target!r} occurs more than once."
        raise UnsupportedIsolationError(message)

    return AnalysedTree(
        simplified_tree=simplified,
        target=target,
        profile=profile,
    )


def choose_isolation_plan(analysed: AnalysedTree) -> PlannedTree:
    """Choose whole-tree normalisation before inverse-rule isolation.

    The only planning decision here is whether the relation needs to be swapped
    so that the target-containing expression is on the left. Already-isolated
    expressions are handled as the base case of the inverse-rule isolator.
    """
    return PlannedTree(
        simplified_tree=analysed.simplified_tree,
        target=analysed.target,
        profile=analysed.profile,
        plan=IsolationPlan(swap_sides=analysed.profile.rhs_count > 0),
    )
