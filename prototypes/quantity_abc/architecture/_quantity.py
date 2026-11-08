"""Concrete generic scalar quantity, independent of unit-bearing tensors."""

from typing import Self, cast

from bilt.units._conversion import conversion_between
from bilt.units._errors import BiltUnitError
from bilt.units._units import Unit

from ._tensor import Tensor


class Quantity[M: float | Tensor = float | Tensor]:
    """Generic quantity contract, retaining the existing scalar constructor.

    A bare Quantity can be used at a model boundary. A Quantity[Tensor]
    describes a unit-bearing tensor without allowing users to instantiate the
    concrete scalar class with a tensor; QuantityTensor provides that object.
    """

    def __init__(self, value: float, unit: Unit) -> None:
        self._magnitude = value
        self._unit = unit

    @property
    def magnitude(self) -> M:
        """Return the scalar or tensor without its unit."""
        return cast("M", self._magnitude)

    @property
    def value(self) -> M:
        """Keep the existing scalar Quantity.value convention."""
        return self.magnitude

    @property
    def unit(self) -> Unit:
        """Return the physical unit."""
        return self._unit

    def _with_magnitude(self, magnitude: float | Tensor, unit: Unit) -> Self:
        if isinstance(magnitude, Tensor):
            message = "Use QuantityTensor for tensor-valued quantities"
            raise TypeError(message)
        return self.__class__(float(magnitude), unit)

    def cast_to(self, unit: Unit) -> Self:
        """Change only compatible unit labels."""
        if self.unit != unit and self.unit.strip_metadata() != unit.strip_metadata():
            message = f"Cannot cast quantity with unit {self.unit} to {unit}"
            raise BiltUnitError(message)
        return self._with_magnitude(self.magnitude, unit)

    def convert_to(self, unit: Unit) -> Self:
        """Convert either magnitude using the existing unit conversion rules."""
        conversion = conversion_between(self.unit, unit)
        magnitude = self.magnitude
        if isinstance(magnitude, Tensor):
            converted = Tensor(magnitude.values * conversion.scale + conversion.offset, magnitude.domain)
        else:
            converted = float(magnitude) * conversion.scale + conversion.offset
        return self._with_magnitude(converted, unit)

    def __add__(self, other: Quantity) -> Self:
        """Demonstrate basic unit-checked arithmetic on a shared contract.

        Only equal units and matching tensor domains are supported here. The
        full Bilt rules (including absolute units and broadcasts) are not
        reimplemented in this architectural experiment.
        """
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.unit != other.unit:
            message = "Prototype addition requires the same unit"
            raise BiltUnitError(message)
        left, right = self.magnitude, other.magnitude
        if isinstance(left, Tensor) and isinstance(right, Tensor):
            if left.domain != right.domain:
                message = "Prototype addition requires identical tensor domains"
                raise ValueError(message)
            result: float | Tensor = Tensor(left.values + right.values, left.domain)
        elif not isinstance(left, Tensor) and not isinstance(right, Tensor):
            result = float(left) + float(right)
        else:
            message = "Mixed scalar/tensor arithmetic belongs to Bilt's lifting machinery"
            raise NotImplementedError(message)
        return self._with_magnitude(result, self.unit)

    def __mul__(self, other: float) -> Self:
        """Scale a quantity without changing its physical unit."""
        if not isinstance(other, float | int):
            return NotImplemented
        magnitude = self.magnitude
        if isinstance(magnitude, Tensor):
            result: float | Tensor = Tensor(magnitude.values * other, magnitude.domain)
        else:
            result = float(magnitude) * other
        return self._with_magnitude(result, self.unit)

    def __rmul__(self, other: float) -> Self:
        """Support a scalar multiplied by a quantity."""
        return self * other


class ScalarQuantity(Quantity[float]):
    """Optional explicit scalar subtype; not required for scalar construction."""
