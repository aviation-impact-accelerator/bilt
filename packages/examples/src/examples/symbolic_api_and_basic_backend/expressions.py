from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .units import DIMENSIONLESS, format_dimension, format_unit, is_canonical_dimensionless_unit, is_unitful_scalar, require_compatible_units, require_dimensionless, unit_divide, unit_multiply, unit_power, unitful_magnitude, unitful_unit

def as_expr(value: Any) -> 'Expr':
    if isinstance(value, Expr): return value
    if is_unitful_scalar(value): return Constant(unitful_magnitude(value), unitful_unit(value))
    return Constant(float(value), DIMENSIONLESS)

class Expr:
    def eval(self, values: dict[str, float]) -> float: raise NotImplementedError
    def unit(self) -> Any: raise NotImplementedError
    def contains(self, symbol_name: str) -> bool: raise NotImplementedError
    def symbols(self) -> set[str]: raise NotImplementedError
    def __add__(self, other: Any) -> 'Expr': return Binary("+", self, as_expr(other))
    def __radd__(self, other: Any) -> 'Expr': return Binary("+", as_expr(other), self)
    def __sub__(self, other: Any) -> 'Expr': return Binary("-", self, as_expr(other))
    def __rsub__(self, other: Any) -> 'Expr': return Binary("-", as_expr(other), self)
    def __mul__(self, other: Any) -> 'Expr': return Binary("*", self, as_expr(other))
    def __rmul__(self, other: Any) -> 'Expr': return Binary("*", as_expr(other), self)
    def __truediv__(self, other: Any) -> 'Expr': return Binary("/", self, as_expr(other))
    def __rtruediv__(self, other: Any) -> 'Expr': return Binary("/", as_expr(other), self)
    def __pow__(self, other: Any) -> 'Expr': return Binary("**", self, as_expr(other))
    def __rpow__(self, other: Any) -> 'Expr': return Binary("**", as_expr(other), self)
    def __neg__(self) -> 'Expr': return Constant(0.0, self.unit()) - self
    def __eq__(self, other: Any):  # type: ignore[override]
        from .relations import Equality
        return Equality(self, as_expr(other))
    def __le__(self, other: Any):
        from .relations import LessEqual
        return LessEqual(self, as_expr(other))
    def __ge__(self, other: Any):
        from .relations import GreaterEqual
        return GreaterEqual(self, as_expr(other))

@dataclass(frozen=True)
class Constant(Expr):
    value: float
    unit_value: Any = DIMENSIONLESS
    def eval(self, values: dict[str, float]) -> float: return self.value
    def unit(self) -> Any: return self.unit_value
    def contains(self, symbol_name: str) -> bool: return False
    def symbols(self) -> set[str]: return set()
    def __str__(self) -> str: return f"{self.value:g}" if is_canonical_dimensionless_unit(self.unit_value) else f"{self.value:g} [{self.unit_value}]"

@dataclass(frozen=True)
class Symbol(Expr):
    name: str
    unit_value: Any = DIMENSIONLESS
    display_name: str | None = None
    def eval(self, values: dict[str, float]) -> float:
        try: return float(values[self.name])
        except KeyError: raise KeyError(f"Missing value for symbol {self.name!r}") from None
    def unit(self) -> Any: return self.unit_value
    def contains(self, symbol_name: str) -> bool: return self.name == symbol_name
    def symbols(self) -> set[str]: return {self.name}
    def __str__(self) -> str: return self.display_name or self.name

@dataclass(frozen=True)
class Binary(Expr):
    op: str
    left: Expr
    right: Expr
    def eval(self, values: dict[str, float]) -> float:
        a, b = self.left.eval(values), self.right.eval(values)
        if self.op == "+": return a + b
        if self.op == "-": return a - b
        if self.op == "*": return a * b
        if self.op == "/": return a / b
        if self.op == "**": return a ** b
        raise ValueError(f"Unknown operator {self.op!r}")
    def unit(self) -> Any:
        lu, ru = self.left.unit(), self.right.unit()
        if self.op in {"+", "-"}: require_compatible_units(lu, ru); return lu
        if self.op == "*": return unit_multiply(lu, ru)
        if self.op == "/": return unit_divide(lu, ru)
        if self.op == "**":
            require_dimensionless(ru)
            if not isinstance(self.right, Constant): require_dimensionless(lu); return lu
            return unit_power(lu, self.right.value)
        raise ValueError(f"Unknown operator {self.op!r}")
    def contains(self, symbol_name: str) -> bool: return self.left.contains(symbol_name) or self.right.contains(symbol_name)
    def symbols(self) -> set[str]: return self.left.symbols() | self.right.symbols()
    def __str__(self) -> str: return f"({self.left} {self.op} {self.right})"

def expression_unit_report(label: str, expr: Expr) -> str:
    unit = expr.unit()
    return f"{label}: {expr}\n  unit:      {format_unit(unit)}\n  dimension: {format_dimension(unit)}"
