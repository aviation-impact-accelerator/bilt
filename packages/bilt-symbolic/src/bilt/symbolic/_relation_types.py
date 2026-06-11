from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
