import functools
import typing

from ._category import UnitCategory
from ._errors import DimensionalityError, UnitError

if typing.TYPE_CHECKING:
    from ._units import Unit


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
        if (self.unit != unit) and (self.unit.strip_metadata() != unit.strip_metadata()):
            message = f"Cannot cast quantity {self!s} to unit {unit!s} as the two units are not equivalent"
            raise UnitError(message)
        return self.__class__(self.value, unit)

    def convert_to(self, unit: Unit) -> Quantity:  # noqa: C901, PLR0912
        match (self.unit.category, unit.category):
            case (UnitCategory.ABSOLUTE, UnitCategory.ABSOLUTE):
                if self.unit is unit:
                    return self

                if self.unit.dimension != unit.dimension:
                    message = (
                        f"Cannot convert quantity with unit dimensionality {self.unit.dimension!s} "
                        f"to unit {unit!s} with dimensionality {unit.dimension!s} as they are "
                        f"different"
                    )
                    raise DimensionalityError(message)

                def create_unit_stack(unit: Unit) -> list[Unit]:
                    stack = [unit]
                    relative_unit = unit.relation.composition[0].unit
                    if relative_unit.is_si_base_unit:
                        return stack
                    if unit.offset is None:
                        message = (
                            f"Cannot build conversion chain for absolute unit {unit!s} because it "
                            f"is not defined directly in terms of an SI base unit but has no offset "
                            f"(zero point) set; absolute units defined relative to a non-SI-base "
                            f"unit must supply a zero point to anchor the chain back to an SI base "
                            f"unit"
                        )
                        raise UnitError(message)
                    return create_unit_stack(unit.offset.composition[0].unit) + stack

                start_unit_stack = create_unit_stack(self.unit)
                end_unit_stack = create_unit_stack(unit)

                if start_unit_stack[0] is not end_unit_stack[0]:
                    message = (
                        f"Cannot convert quantity with unit {self} to unit {unit} as the two do "
                        f"not share a common root unit in their definition"
                    )
                    raise UnitError(message)

                index = 0
                for start_unit, end_unit in zip(start_unit_stack, end_unit_stack):  # noqa: B905
                    if start_unit != end_unit:
                        break
                    index += 1

                value = self.value

                for via_unit in reversed(start_unit_stack[index:]):
                    scale = via_unit.relation.composition[0].unit.relation.scale
                    value *= scale
                    offset = via_unit.offset
                    value += 0.0 if offset is None else offset.scale

                for via_unit in end_unit_stack[index:]:
                    offset = via_unit.offset
                    value -= 0.0 if offset is None else offset.scale
                    scale = via_unit.relation.composition[0].unit.relation.scale
                    value /= scale

                return Quantity(value, unit)
            case (_, UnitCategory.ABSOLUTE):
                message = (
                    f"Cannot convert quantity with unit {self} to quantity with absolute unit "
                    f"{unit} because relative and absolute units cannot be interconverted"
                )
                raise UnitError(message)
            case (UnitCategory.ABSOLUTE, _):
                message = (
                    f"Cannot convert quantity with absolute unit {self} to quantity with unit "
                    f"{unit} because relative and absolute units cannot be interconverted"
                )
                raise UnitError(message)

            case _:
                if self.unit.dimension != unit.dimension:
                    message = (
                        f"Cannot convert quantity with unit dimensionality {self.unit.dimension!s} "
                        f"to unit {unit!s} with dimensionality {unit.dimension!s} as they are"
                        f"different"
                    )
                    raise DimensionalityError(message)
                if self.unit.metadata is not unit.metadata:
                    message = (
                        f"Cannot convert quantity with unit with metadata {self.unit.metadata!s} "
                        f"to unit {unit!s} with metadata {unit.metadata!s} as they are different"
                    )
                    raise UnitError(message)

                scale = self.unit.to_si_base_units().relation.scale / unit.to_si_base_units().relation.scale

                return Quantity(scale * self.value, unit)

    def strip_metadata(self) -> typing.Self:
        return self.__class__(self.value, self.unit.strip_metadata())

    def __pos__(self) -> typing.Self:
        return self.__class__(+self.value, self.unit)

    def __neg__(self) -> typing.Self:
        return self.__class__(-self.value, self.unit)

    def __abs__(self) -> typing.Self:
        return self.__class__(abs(self.value), self.unit)

    def __add__(self, other: typing.Self) -> typing.Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.unit.category is UnitCategory.ABSOLUTE or other.unit.category is UnitCategory.ABSOLUTE:
            if self.unit.category is UnitCategory.ABSOLUTE and other.unit.category is UnitCategory.ABSOLUTE:
                message = f"Cannot add two quantities with absolute units of {self.unit!s} and {other.unit!s}"
                raise UnitError(message)
        elif self.unit is not other.unit:
            if self.unit.dimension != other.unit.dimension:
                message = (
                    f"Cannot add quantity {self!s} with unit {self.unit!s} and dimensionality "
                    f"{self.unit.dimension!s} to quantity {other!s} with unit {other.unit!s} and "
                    f"{other.unit.dimension!s} as they are dimensionally inconsistent"
                )
                raise DimensionalityError(message)
            if self.unit != other.unit and self.unit.metadata != other.unit.metadata:
                message = (
                    f"Cannot add quantity {self!s} with unit {self.unit!s} and metadata "
                    f"{self.unit.metadata!s} to quantity {other!s} with unit {other.unit!s} and "
                    f"metadata {other.unit.metadata!s} because the units do not have consistent "
                    f"metadata"
                )
                raise UnitError(message)
            if self.unit != other.unit:
                message = (
                    f"Cannot add quantity {self!s} with unit {self.unit!s} to quantity {other!s} "
                    f"with unit {other.unit!s} as the units are inconsistent, use the "
                    f"`Quantity.convert_to()` method on one of the operands to convert them to "
                    f"consistent units"
                )
                raise UnitError(message)
            message = (
                f"Cannot add quantity {self!s} with unit {self.unit!s} to quantity {other!s} with "
                f"unit {other.unit!s} as the units are inconsistent, use the `Quantity.cast_to()` "
                f"method on one of the operands to cast them to consistent units"
            )
            raise UnitError(message)
        return self.__class__(self.value + other.value, self.unit)

    def __radd__(self, other: int) -> Quantity:
        if not isinstance(other, int) and other != 0:
            return NotImplemented
        return self

    def __sub__(self, other: typing.Self) -> Quantity:
        if self.unit.category is UnitCategory.ABSOLUTE and other.unit.category is UnitCategory.ABSOLUTE:
            return (self.value * self.unit.relation.composition[0].unit) + (
                -other.value * other.unit.relation.composition[0].unit
            )
        return self + (-other)

    def __mul__(self, other: typing.Self | float) -> typing.Self:
        if isinstance(other, self.__class__):
            return self.__class__(self.value * other.value, self.unit * other.unit)
        if isinstance(other, float):
            return self.__class__(self.value * other, self.unit)
        return NotImplemented

    def __rmul__(self, other: float) -> typing.Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(self.value * other, self.unit)

    def __truediv__(self, other: typing.Self | float) -> typing.Self:
        if isinstance(other, self.__class__):
            return self.__class__(self.value / other.value, self.unit / other.unit)
        if isinstance(other, float):
            return self.__class__(self.value / other, self.unit)
        return NotImplemented

    def __rtruediv__(self, other: float) -> typing.Self:
        if not isinstance(other, float):
            return NotImplemented
        return self.__class__(other / self.value, self.unit**-1)

    def __pow__(self, other: float | typing.Self) -> typing.Self:
        match other:
            case int():
                return self.__class__(self.value**other, self.unit**other)
            case float() if self.unit.is_dimensionless:
                return self.__class__(self.value**other, self.unit)
            case Quantity() if other.unit.is_dimensionless and other.unit.relation.scale == 1.0:
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
        if self.unit != other.unit:
            message = (
                f"Cannot compare quantity {self!s} with unit {self.unit!s} to quantity {other!s} "
                f"with unit {other.unit!s}"
            )
            raise UnitError(message)
        return self.value < other.value

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return f"{self.value!s} {self.unit!s}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value!r}, unit={self.unit!r})"
