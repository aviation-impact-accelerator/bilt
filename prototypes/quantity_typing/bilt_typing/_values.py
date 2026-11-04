"""Experimental quantity wrapper; the adjacent stub describes supported arithmetic."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
import numpy as np
from bilt import Tensor as NativeTensor, tensor as native_tensor
from bilt.units import Quantity as NativeQuantity, Unit as NativeUnit
from bilt.symbolic._domains import Domain
from ._markers import DimensionType

D = TypeVar("D", bound=DimensionType)
R = TypeVar("R")
T = TypeVar("T")


@dataclass(frozen=True)
class Tensor(Generic[T]):
    values: np.ndarray
    domain: Domain

    def __post_init__(self):
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))
        if self.values.ndim != len(self.domain.dims):
            raise ValueError("Tensor rank does not match its domain")


@dataclass(frozen=True)
class Unit(Generic[D]):
    runtime: NativeUnit
    dimension: type[D]

    def __post_init__(self):
        from bilt.units._category import UnitCategory

        if self.runtime.dimension is not self.dimension.runtime:
            raise TypeError("Unit and static dimension disagree")
        if (self.runtime.category is UnitCategory.ABSOLUTE) != self.dimension.absolute:
            raise TypeError("Unit and static absolute marker disagree")

    def __rmul__(self, value):
        if not isinstance(value, (float, int)):
            return NotImplemented
        return Quantity(float(value), self)


class Quantity(Generic[D, R]):
    def __init__(self, value, unit):
        if not isinstance(value, (float, Tensor)):
            raise TypeError("Expected float or Tensor magnitude")
        self.value = value
        self.unit = unit

    @property
    def domain(self):
        return self.value.domain if isinstance(self.value, Tensor) else Domain.scalar()

    def convert_to(self, unit):
        if self.unit.dimension is not unit.dimension:
            raise TypeError("Incompatible physical dimensions or absolute semantics")
        raw = self.value.values if isinstance(self.value, Tensor) else self.value
        converted = NativeQuantity(raw, self.unit.runtime).convert_to(unit.runtime).value
        return Quantity(Tensor(converted, self.domain) if isinstance(self.value, Tensor) else float(converted), unit)

    def _binary(self, other, op):
        if not isinstance(other, Quantity):
            return NotImplemented
        if isinstance(self.value, Tensor) and isinstance(other.value, Tensor):
            if self.domain != other.domain or self.value.values.shape != other.value.values.shape:
                raise ValueError("Direct prototype arithmetic requires matching tensor domains and shapes")
        left = self.value.values if isinstance(self.value, Tensor) else self.value
        right = other.value.values if isinstance(other.value, Tensor) else other.value
        if op == "add":
            converted = other.convert_to(self.unit)
            right = converted.value.values if isinstance(converted.value, Tensor) else converted.value
            result = NativeQuantity(left, self.unit.runtime) + NativeQuantity(right, self.unit.runtime)
        elif op == "mul":
            result = NativeQuantity(left, self.unit.runtime) * NativeQuantity(right, other.unit.runtime)
        else:
            result = NativeQuantity(left, self.unit.runtime) / NativeQuantity(right, other.unit.runtime)
        from .dimension_types import marker_for

        marker = marker_for(result.unit)
        magnitude = result.value
        if isinstance(self.value, Tensor) or isinstance(other.value, Tensor):
            domain = self.domain if isinstance(self.value, Tensor) else other.domain
            magnitude = Tensor(magnitude, domain)
        else:
            magnitude = float(magnitude)
        return Quantity(magnitude, Unit(result.unit, marker))

    def __add__(self, other):
        return self._binary(other, "add")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            if self.unit.dimension.absolute:
                raise TypeError("Scaling absolute quantities is not supported by this prototype")
            magnitude = (
                Tensor(self.value.values * other, self.domain) if isinstance(self.value, Tensor) else self.value * other
            )
            return Quantity(magnitude, self.unit)
        return self._binary(other, "mul")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return self * (1.0 / other)
        return self._binary(other, "div")

    def __eq__(self, other):
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.unit.dimension is not other.unit.dimension or self.domain != other.domain:
            return False
        converted = other.convert_to(self.unit)
        left = self.value.values if isinstance(self.value, Tensor) else self.value
        right = converted.value.values if isinstance(converted.value, Tensor) else converted.value
        return bool(np.array_equal(left, right))

    def sum(self, axis):
        if not isinstance(self.value, Tensor):
            raise TypeError("sum requires tensor data")
        if self.unit.dimension.absolute:
            raise TypeError("Absolute quantities cannot be summed in this prototype")
        index = self.domain.dims.index(axis)
        result = self.value.values.sum(axis=index)
        remaining = tuple(d for d in self.domain.dims if d != axis)
        magnitude = Tensor(result, Domain(remaining)) if remaining else float(result)
        return Quantity(magnitude, self.unit)

    def __repr__(self):
        return f"Quantity({self.value!r}, {self.unit.runtime!s})"


def tensor(values, *, over, unit):
    native = native_tensor(values, over=over)
    return Quantity(Tensor(native.values, native.domain), unit)


def to_native(value):
    if isinstance(value, Quantity):
        if isinstance(value.value, Tensor):
            return NativeTensor(value.value.values, value.value.domain, value.unit.runtime)
        return NativeQuantity(value.value, value.unit.runtime)
    return value


def from_native(value):
    from .dimension_types import marker_for

    if isinstance(value, NativeTensor) and value.unit is not None:
        return Quantity(Tensor(value.values, value.domain), Unit(value.unit, marker_for(value.unit)))
    if isinstance(value, NativeQuantity):
        return Quantity(float(value.value), Unit(value.unit, marker_for(value.unit)))
    return value
