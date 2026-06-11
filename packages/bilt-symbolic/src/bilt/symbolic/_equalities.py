from dataclasses import dataclass
from typing import TYPE_CHECKING

__all__ = [
    "Equality",
]

if TYPE_CHECKING:
    from bilt.symbolic._expressions import Expr


@dataclass(frozen=True, slots=True)
class Equality:
    lhs: Expr
    rhs: Expr

    def __bool__(self) -> bool:
        msg = "Equality objects cannot be used as booleans."
        raise TypeError(msg)
