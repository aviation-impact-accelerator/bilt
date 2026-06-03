from __future__ import annotations

import typing
from dataclasses import dataclass

from bilt.symbolic._coercion import as_expr

__all__ = [
    "Relation",
    "equals",
    "relation_to_string",
]
if typing.TYPE_CHECKING:
    from bilt.symbolic._expressions import Expr


@dataclass(frozen=True, slots=True)
class Relation:
    lhs: Expr
    rhs: Expr

    def __bool__(self) -> bool:
        msg = "Relation objects cannot be used as booleans."
        raise TypeError(msg)

    def __str__(self) -> str:
        return relation_to_string(self)


def equals(lhs: object, rhs: object) -> Relation:
    return Relation(
        lhs=as_expr(lhs),
        rhs=as_expr(rhs),
    )


def relation_to_string(relation: Relation) -> str:
    return f"{relation.lhs} == {relation.rhs}"
