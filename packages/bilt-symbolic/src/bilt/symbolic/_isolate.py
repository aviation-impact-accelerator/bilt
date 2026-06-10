from typing import TYPE_CHECKING

from bilt.symbolic._errors import IsolationError
from bilt.symbolic._peel import SymbolTarget, peel
from bilt.symbolic._relations import IsolatedRelation, Relation
from bilt.symbolic._walk import count_occurrences

if TYPE_CHECKING:
    from bilt.symbolic._expressions import Expr


def isolate(relation: Relation, target: Expr) -> IsolatedRelation:
    """Structurally isolate target in a symbolic relation.

    This is deliberately limited. It assumes the target:
    - is a single symbolic variable
    - occurs exactly once
    - occurs on exactly one side of the relation
    - can be isolated using the supported peel rules
    """
    symbol_target = SymbolTarget.from_expr(target)

    lhs = relation.lhs
    rhs = relation.rhs

    lhs_count = count_occurrences(lhs.rep, symbol_target.rep)
    rhs_count = count_occurrences(rhs.rep, symbol_target.rep)

    total_count = lhs_count + rhs_count

    if total_count == 0:
        msg = "Target {target.rep} does not occur in relation {relation.rep}."
        raise IsolationError(msg)

    if total_count > 1:
        msg = "Target {target.rep} within relation {relation.rep} occurs more than once; structural isolation cannot handle this."
        raise IsolationError(msg)

    if rhs_count == 1:
        lhs, rhs = rhs, lhs

    isolated_rhs = peel(
        lhs=lhs,
        rhs=rhs,
        target=symbol_target,
    )

    return IsolatedRelation(lhs=symbol_target.expr, rhs=isolated_rhs)
