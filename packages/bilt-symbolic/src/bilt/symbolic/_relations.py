from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bilt.symbolic._expressions import Expr, as_expr

class Relation:
    lhs: Expr
    rhs: Expr

    @property
    def kind(self) -> str:
        return self.__class__.__name__

    def residual(self) -> Expr:
        raise NotImplementedError

    def check_units(self) -> None:
        require_compatible_units(self.lhs.unit(), self.rhs.unit())

    def unit_analysis(self, *, name: str = "<relation>") -> UnitAnalysis:
        lu, ru = self.lhs.unit(), self.rhs.unit()
        return UnitAnalysis(name, str(self), self.lhs, self.rhs, lu, ru, units_compatible(lu, ru), self.kind)

    def unit_report(self, *, name: str = "<relation>") -> str:
        return self.unit_analysis(name=name).to_text()

    def symbols(self) -> set[str]:
        return self.lhs.symbols() | self.rhs.symbols()


@dataclass(frozen=True)
class Equality(Relation):
    lhs: Expr
    rhs: Expr

    @property
    def kind(self) -> str:
        return "equality"

    def residual(self) -> Expr:
        return self.lhs - self.rhs

    def __str__(self) -> str:
        return f"{self.lhs} == {self.rhs}"


@dataclass(frozen=True)
class LessEqual(Relation):
    lhs: Expr
    rhs: Expr

    @property
    def kind(self) -> str:
        return "less_equal"

    def residual(self) -> Expr:
        return self.lhs - self.rhs

    def __str__(self) -> str:
        return f"{self.lhs} <= {self.rhs}"


@dataclass(frozen=True)
class GreaterEqual(Relation):
    lhs: Expr
    rhs: Expr

    @property
    def kind(self) -> str:
        return "greater_equal"

    def residual(self) -> Expr:
        return self.lhs - self.rhs

    def __str__(self) -> str:
        return f"{self.lhs} >= {self.rhs}"


def Eq(lhs: Any, rhs: Any) -> Equality:
    return Equality(as_expr(lhs), as_expr(rhs))


def Le(lhs: Any, rhs: Any) -> LessEqual:
    return LessEqual(as_expr(lhs), as_expr(rhs))


def Ge(lhs: Any, rhs: Any) -> GreaterEqual:
    return GreaterEqual(as_expr(lhs), as_expr(rhs))
