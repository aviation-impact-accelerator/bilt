from __future__ import annotations
from dataclasses import dataclass
from typing import Any

class UnitError(ValueError):
    pass

@dataclass(frozen=True)
class DimensionlessUnit:
    name: str = "dimensionless"
    @property
    def is_dimensionless(self) -> bool: return True
    def __mul__(self, other: Any) -> Any: return other
    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (int, float)): return UnitfulScalar(float(other), self)
        return other
    def __truediv__(self, other: Any) -> Any:
        if is_dimensionless_unit(other): return self
        return UnitExpression("/", self, other)
    def __rtruediv__(self, other: Any) -> Any:
        if isinstance(other, (int, float)): return UnitfulScalar(float(other), self)
        return UnitExpression("/", other, self)
    def __pow__(self, exponent: float) -> Any: return self
    def __str__(self) -> str: return self.name

DIMENSIONLESS = DimensionlessUnit()

@dataclass(frozen=True)
class UnitExpression:
    op: str
    left: Any
    right: Any
    def __mul__(self, other: Any) -> Any: return UnitExpression("*", self, other)
    def __truediv__(self, other: Any) -> Any: return UnitExpression("/", self, other)
    def __pow__(self, exponent: float) -> Any: return UnitExpression("**", self, exponent)
    def __str__(self) -> str: return f"({self.left} {self.op} {self.right})"

@dataclass(frozen=True)
class UnitfulScalar:
    magnitude: float
    unit: Any

def is_unitful_scalar(value: Any) -> bool:
    return isinstance(value, UnitfulScalar) or (hasattr(value, "value") and hasattr(value, "unit"))

def unitful_magnitude(value: Any) -> float:
    return float(value.magnitude if isinstance(value, UnitfulScalar) else value.value)

def unitful_unit(value: Any) -> Any:
    return value.unit

def is_dimensionless_unit(unit: Any) -> bool:
    if isinstance(unit, DimensionlessUnit): return True
    flag = getattr(unit, "is_dimensionless", None)
    if isinstance(flag, bool): return flag
    relation = getattr(unit, "relation", None)
    if relation is not None:
        try: return len(relation) == 0
        except TypeError: pass
    return False

def is_canonical_dimensionless_unit(unit: Any) -> bool:
    return unit is DIMENSIONLESS or (is_dimensionless_unit(unit) and getattr(unit, "name", None) in {"DIMENSIONLESS", "dimensionless"})

def unit_dimension(unit: Any) -> Any | None:
    return getattr(unit, "dimension", None)

def units_compatible(left: Any, right: Any) -> bool:
    if is_dimensionless_unit(left) and is_dimensionless_unit(right): return True
    ld, rd = unit_dimension(left), unit_dimension(right)
    if ld is not None and rd is not None: return ld == rd
    if left == right: return True
    for name in ("is_compatible_with", "compatible_with"):
        method = getattr(left, name, None)
        if callable(method) and bool(method(right)): return True
        method = getattr(right, name, None)
        if callable(method) and bool(method(left)): return True
    return False

def require_compatible_units(left: Any, right: Any) -> None:
    if not units_compatible(left, right): raise UnitError(f"Incompatible units: {left!s} and {right!s}")

def require_dimensionless(unit: Any) -> None:
    if not is_dimensionless_unit(unit): raise UnitError(f"Expected dimensionless unit, got {unit!s}")

def unit_multiply(left: Any, right: Any) -> Any:
    if is_dimensionless_unit(left): return right
    if is_dimensionless_unit(right): return left
    return left * right

def unit_divide(left: Any, right: Any) -> Any:
    if is_dimensionless_unit(right): return left
    if is_dimensionless_unit(left): return 1 / right
    return left / right

def unit_power(unit: Any, exponent: float) -> Any:
    if is_dimensionless_unit(unit): return unit
    return unit ** int(exponent) if float(exponent).is_integer() else unit ** exponent

def format_unit(unit: Any) -> str:
    try: return str(unit)
    except Exception: return repr(unit)

def format_dimension(unit: Any) -> str:
    dim = unit_dimension(unit)
    if dim is None: return "DIMENSIONLESS" if is_dimensionless_unit(unit) else "<unknown dimension>"
    try: return str(dim)
    except Exception: return repr(dim)

@dataclass(frozen=True)
class SimpleUnit:
    dims: tuple[tuple[str, float], ...]
    name: str | None = None
    @staticmethod
    def base(name: str) -> 'SimpleUnit': return SimpleUnit(((name, 1.0),), name=name)
    @staticmethod
    def dimensionless() -> 'SimpleUnit': return SimpleUnit(tuple(), name="dimensionless")
    @property
    def is_dimensionless(self) -> bool: return not self.dims
    @property
    def dimension(self) -> tuple[tuple[str, float], ...]: return self.dims
    def _dims_dict(self) -> dict[str, float]: return dict(self.dims)
    @classmethod
    def _from_dict(cls, dims: dict[str, float], name: str | None = None) -> 'SimpleUnit':
        return cls(tuple(sorted((k, v) for k, v in dims.items() if abs(v) > 1e-12)), name=name)
    def __mul__(self, other: Any) -> Any:
        if isinstance(other, (int, float)): return UnitfulScalar(float(other), self)
        if is_dimensionless_unit(other): return self
        if not isinstance(other, SimpleUnit): return UnitExpression("*", self, other)
        dims = self._dims_dict()
        for k, v in other.dims: dims[k] = dims.get(k, 0.0) + v
        return SimpleUnit._from_dict(dims)
    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, (int, float)): return UnitfulScalar(float(other), self)
        return self.__mul__(other)
    def __truediv__(self, other: Any) -> Any:
        if is_dimensionless_unit(other): return self
        if not isinstance(other, SimpleUnit): return UnitExpression("/", self, other)
        dims = self._dims_dict()
        for k, v in other.dims: dims[k] = dims.get(k, 0.0) - v
        return SimpleUnit._from_dict(dims)
    def __rtruediv__(self, other: Any) -> Any:
        if isinstance(other, (int, float)): return UnitfulScalar(float(other), SimpleUnit.dimensionless() / self)
        if is_dimensionless_unit(other): return SimpleUnit.dimensionless() / self
        return UnitExpression("/", other, self)
    def __pow__(self, exponent: float) -> 'SimpleUnit': return SimpleUnit._from_dict({k: v * exponent for k, v in self.dims})
    def __str__(self) -> str:
        if self.name: return self.name
        if not self.dims: return "dimensionless"
        return " ".join(f"{k}^{v:g}" for k, v in self.dims)
