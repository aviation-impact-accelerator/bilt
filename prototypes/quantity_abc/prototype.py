"""Experimental shared quantity contract; not a replacement for bilt.Quantity."""

from abc import ABC, abstractmethod
from typing import Self

from bilt import Tensor
from bilt.units import Unit
from bilt.units._conversion import conversion_between
from bilt.units._errors import BiltUnitError


class Quantity[M: float | Tensor = float | Tensor](ABC):
    """A quantity with either a scalar or tensor magnitude.

    The default type argument deliberately permits either representation.
    Physical dimensions remain Annotated metadata, not generic parameters.
    """

    @property
    @abstractmethod
    def magnitude(self) -> M:
        """Return the value without its physical unit."""

    @property
    @abstractmethod
    def unit(self) -> Unit:
        """Return the quantity's (non-optional) unit."""

    @abstractmethod
    def _with_unit(self, unit: Unit) -> Self:
        """Return the same magnitude labelled with another equivalent unit."""

    @abstractmethod
    def _converted(self, scale: float, offset: float, unit: Unit) -> Self:
        """Apply a conversion while preserving scalar/tensor representation."""

    def cast_to(self, unit: Unit) -> Self:
        """Change equivalent unit metadata without changing the magnitude."""
        if self.unit != unit and self.unit.strip_metadata() != unit.strip_metadata():
            message = f"Cannot cast quantity with unit {self.unit} to non-equivalent unit {unit}"
            raise BiltUnitError(message)
        return self._with_unit(unit)

    def convert_to(self, unit: Unit) -> Self:
        """Convert values using Bilt's existing unit conversion rules."""
        conversion = conversion_between(self.unit, unit)
        return self._converted(conversion.scale, conversion.offset, unit)


class ScalarQuantity(Quantity[float]):
    """Scalar implementation of the shared quantity contract."""

    __slots__ = ("_magnitude", "_unit")

    def __init__(self, magnitude: float, unit: Unit) -> None:
        self._magnitude = magnitude
        self._unit = unit

    @property
    def magnitude(self) -> float:
        return self._magnitude

    @property
    def unit(self) -> Unit:
        return self._unit

    def _with_unit(self, unit: Unit) -> Self:
        return self.__class__(self.magnitude, unit)

    def _converted(self, scale: float, offset: float, unit: Unit) -> Self:
        return self.__class__(self.magnitude * scale + offset, unit)


class QuantityTensor(Tensor, Quantity[Tensor]):
    """A unit-bearing Bilt tensor that is also a Quantity.

    This deliberately uses Bilt's existing Tensor superclass. Re-declaring
    the unit type and checking it at runtime prevents unitless instances from
    satisfying the concrete quantity contract.
    """

    unit: Unit

    def __post_init__(self) -> None:
        Tensor.__post_init__(self)
        if self.unit is None:
            message = "QuantityTensor requires a unit; use Tensor for unitless values"
            raise BiltUnitError(message)

    @property
    def magnitude(self) -> Tensor:
        return Tensor(values=self.values, domain=self.domain)

    def convert_to(self, unit: Unit) -> Self:
        """Disambiguate Tensor and Quantity conversion return types."""
        return Quantity.convert_to(self, unit)

    def _with_unit(self, unit: Unit) -> Self:
        return self.__class__(values=self.values, domain=self.domain, unit=unit)

    def _converted(self, scale: float, offset: float, unit: Unit) -> Self:
        return self.__class__(values=self.values * scale + offset, domain=self.domain, unit=unit)
