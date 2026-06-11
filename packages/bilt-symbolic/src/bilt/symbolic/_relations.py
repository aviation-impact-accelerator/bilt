from dataclasses import dataclass
from typing import TYPE_CHECKING

from bilt.symbolic._coercion import as_expr
from bilt.symbolic._equalities import Equality

__all__ = [
    "Relation",
    "as_relation",
    "equals",
]

if TYPE_CHECKING:
    from bilt.symbolic._expressions import Expr


@dataclass(frozen=True, slots=True)
class Relation:
    lhs: Expr
    rhs: Expr

    def __bool__(self) -> bool:
        msg = "Relation objects cannot be used as booleans."
        raise TypeError(msg)

    def __str__(self) -> str:
        return f"{self.lhs} == {self.rhs}"


def equals(lhs: object, rhs: object) -> Equality:
    return Equality(
        lhs=as_expr(lhs),
        rhs=as_expr(rhs),
    )


def as_relation(value: object) -> Relation:
    if isinstance(value, Relation):
        return value

    if isinstance(value, Equality):
        return Relation(lhs=value.lhs, rhs=value.rhs)

    msg = f"Cannot convert {type(value).__name__} to Relation."
    raise TypeError(msg)
