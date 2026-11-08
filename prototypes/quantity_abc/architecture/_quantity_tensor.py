"""Composition layer: only this module needs both tensor and quantity APIs."""

from typing import Self, overload

import numpy as np

from bilt.units._errors import BiltUnitError
from bilt.units._units import Unit

from ._quantity import Quantity, ScalarQuantity
from ._tensor import Tensor


class QuantityTensor(Tensor, Quantity[Tensor]):
    """Unit-bearing tensor satisfying both isinstance checks."""

    _quantity_unit: Unit

    def __init__(self, values: object, domain: tuple[str, ...], unit: Unit) -> None:
        Tensor.__init__(self, np.asarray(values), domain)
        if not isinstance(unit, Unit):
            message = "QuantityTensor requires a physical unit"
            raise BiltUnitError(message)
        object.__setattr__(self, "_quantity_unit", unit)

    @property
    def unit(self) -> Unit:
        """Return the non-optional tensor unit."""
        return self._quantity_unit

    @property
    def magnitude(self) -> Tensor:
        """Expose a unitless tensor with identical values and domain."""
        return Tensor(self.values, self.domain)

    @property
    def value(self) -> Tensor:
        """Return unitless magnitude, matching the generic contract."""
        return self.magnitude

    def _with_magnitude(self, magnitude: Tensor, unit: Unit) -> Self:
        return self.__class__(magnitude.values, magnitude.domain, unit)


class TypedUnit:
    """Temporary operator adapter; production Unit is deliberately unchanged."""

    def __init__(self, unit: Unit) -> None:
        self.native = unit

    @overload
    def __rmul__(self, value: float) -> ScalarQuantity: ...

    @overload
    def __rmul__(self, value: Tensor) -> QuantityTensor: ...

    def __rmul__(self, value: float | Tensor) -> ScalarQuantity | QuantityTensor:
        if isinstance(value, Tensor):
            return QuantityTensor(value.values, value.domain, self.native)
        if isinstance(value, float | int):
            return ScalarQuantity(float(value), self.native)
        return NotImplemented


def typed(unit: Unit) -> TypedUnit:
    """Wrap a production unit for testing the future multiplication API."""
    return TypedUnit(unit)
