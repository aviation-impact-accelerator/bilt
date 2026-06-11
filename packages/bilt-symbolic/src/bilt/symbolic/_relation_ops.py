from __future__ import annotations

from bilt.symbolic._coercion import as_expr
from bilt.symbolic._relation_types import Relation


def equals(lhs: object, rhs: object) -> Relation:
    return Relation(
        lhs=as_expr(lhs),
        rhs=as_expr(rhs),
    )
