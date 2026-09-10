import functools
import typing

from bilt.units._category import UnitCategory
from bilt.units._conversion import conversion_between
from bilt.units._validation import validate_addition, validate_cast, validate_comparison

if typing.TYPE_CHECKING:
    from bilt.units._units import Unit


@typing.final
@functools.total_ordering
class Quantity:
    __slots__ = ("_unit", "_value")

    _value: float
    _unit: Unit

    def __init__(self, value: float, unit: Unit) -> None:
        self._value = value
        self._unit = unit

    @property
    def value(self) -> float:
        return self._value

    @property
    def unit(self) -> Unit:
        return self._unit

    def cast_to(self, unit: Unit) -> typing.Self:
        validate_cast(self, unit)
        return self.__class__(self.value, unit)

    def convert_to(self, unit: Unit) -> Quantity:
        if self.unit.category is UnitCategory.ABSOLUTE and self.unit is unit:
            return self

        conversion = conversion_between(self.unit, unit)

        return Quantity(
            conversion.apply(self.value),
            unit,
        )

    def strip_metadata(self) -> typing.Self:
        return self.__class__(self.value, self.unit.strip_metadata())

    def __pos__(self) -> typing.Self:
        return self.__class__(+self.value, self.unit)

    def __neg__(self) -> typing.Self:
        return self.__class__(-self.value, self.unit)

    def __abs__(self) -> typing.Self:
        return self.__class__(abs(self.value), self.unit)

    def __add__(self, other: Quantity) -> typing.Self:
        if not isinstance(other, self.__class__):
            return NotImplemented

        validate_addition(self, other)

        if self.unit.category is UnitCategory.ABSOLUTE:
            unit = self.unit
        elif other.unit.category is UnitCategory.ABSOLUTE:
            unit = other.unit
        else:
            unit = self.unit

        return self.__class__(
            self.value + other.value,
            unit,
        )

    def __radd__(self, other: int) -> Quantity:
        if not isinstance(other, int) and other != 0:
            return NotImplemented
        return self

    def __sub__(self, other: Quantity) -> Quantity:
        if self.unit.category is UnitCategory.ABSOLUTE and other.unit.category is UnitCategory.ABSOLUTE:
            return (self.value * self.unit.relation.composition[0].unit) + (
                -other.value * other.unit.relation.composition[0].unit
            )
        return self + (-other)

    def __mul__(self, other: Quantity | float) -> typing.Self:
        if isinstance(other, self.__class__):
            return self.__class__(self.value * other.value, self.unit * other.unit)
        if isinstance(other, float):
            return self.__class__(self.value * other, self.unit)
        return NotImplemented

    def __rmul__(self, other: float) -> typing.Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(self.value * other, self.unit)

    def __truediv__(self, other: Quantity | float) -> typing.Self:
        if isinstance(other, self.__class__):
            return self.__class__(self.value / other.value, self.unit / other.unit)
        if isinstance(other, float):
            return self.__class__(self.value / other, self.unit)
        return NotImplemented

    def __rtruediv__(self, other: float) -> typing.Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(other / self.value, self.unit**-1)

    def __pow__(self, other: float | Quantity) -> typing.Self:
        match other:
            case int():
                return self.__class__(self.value**other, self.unit**other)
            case float() if self.unit.is_dimensionless:
                return self.__class__(self.value**other, self.unit)
            case Quantity() if self.unit.is_dimensionless and self.unit.relation.scale == 1.0:
                return self.__class__(self.value**other.value, self.unit)
        return NotImplemented

    def __rpow__(self, other: float) -> typing.Self:
        if self.unit.is_dimensionless and self.unit.relation.scale == 1.0:
            return self.__class__(other**self.value, self.unit)
        return NotImplemented

    def __round__(self, ndigits: int | None = None) -> typing.Self:
        return self.__class__(round(self.value, ndigits=ndigits), self.unit)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self.value == other.value) and (self.unit == other.unit)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        validate_comparison(self, other)
        return self.value < other.value

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return f"{self.value!s} {self.unit!s}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value!r}, unit={self.unit!r})"
