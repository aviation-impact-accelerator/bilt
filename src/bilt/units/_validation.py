import typing

from bilt.units._category import UnitCategory
from bilt.units._errors import DimensionalityError, UnitError

if typing.TYPE_CHECKING:
    from bilt.units._conversion import QuantityLike
    from bilt.units._units import Unit


def validate_cast(quantity: QuantityLike, unit: Unit) -> None:
    if (quantity.unit != unit) and (quantity.unit.strip_metadata() != unit.strip_metadata()):
        message = f"Cannot cast quantity {quantity!s} to unit {unit!s} as the two units are not equivalent"
        raise UnitError(message)


def validate_addition(
    left: QuantityLike,
    right: QuantityLike,
) -> None:
    if left.unit.category is UnitCategory.ABSOLUTE or right.unit.category is UnitCategory.ABSOLUTE:
        if left.unit.category is UnitCategory.ABSOLUTE and right.unit.category is UnitCategory.ABSOLUTE:
            message = f"Cannot add two quantities with absolute units of {left.unit!s} and {right.unit!s}"
            raise UnitError(message)
        if left.unit.dimension != right.unit.dimension:
            message = (
                f"Cannot add quantity {left!s} with unit {left.unit!s} and "
                f"dimensionality {left.unit.dimension!s} to quantity "
                f"{right!s} with unit {right.unit!s} and "
                f"{right.unit.dimension!s} as they are dimensionally "
                f"inconsistent"
            )
            raise DimensionalityError(message)
        if left.unit.metadata != right.unit.metadata:
            message = (
                f"Cannot add quantity {left!s} with unit {left.unit!s} and "
                f"metadata {left.unit.metadata!s} to quantity {right!s} with "
                f"unit {right.unit!s} and metadata {right.unit.metadata!s} "
                f"because the units do not have consistent metadata"
            )
            raise UnitError(message)
    elif left.unit is not right.unit:
        if left.unit.dimension != right.unit.dimension:
            message = (
                f"Cannot add quantity {left!s} with unit {left.unit!s} and "
                f"dimensionality {left.unit.dimension!s} to quantity "
                f"{right!s} with unit {right.unit!s} and "
                f"{right.unit.dimension!s} as they are dimensionally "
                f"inconsistent"
            )
            raise DimensionalityError(message)
        if left.unit != right.unit and left.unit.metadata != right.unit.metadata:
            message = (
                f"Cannot add quantity {left!s} with unit {left.unit!s} and "
                f"metadata {left.unit.metadata!s} to quantity {right!s} with "
                f"unit {right.unit!s} and metadata {right.unit.metadata!s} "
                f"because the units do not have consistent metadata"
            )
            raise UnitError(message)
        if left.unit != right.unit:
            message = (
                f"Cannot add quantity {left!s} with unit {left.unit!s} to "
                f"quantity {right!s} with unit {right.unit!s} as the units are "
                f"inconsistent, use the `Quantity.convert_to()` method on one "
                f"of the operands to convert them to consistent units"
            )
            raise UnitError(message)
        message = (
            f"Cannot add quantity {left!s} with unit {left.unit!s} to quantity "
            f"{right!s} with unit {right.unit!s} as the units are inconsistent, "
            f"use the `Quantity.cast_to()` method on one of the operands to "
            f"cast them to consistent units"
        )
        raise UnitError(message)


def validate_comparison(
    left: QuantityLike,
    right: QuantityLike,
) -> None:
    if left.unit != right.unit:
        message = (
            f"Cannot compare quantity {left!s} with unit {left.unit!s} to quantity {right!s} with unit {right.unit!s}"
        )
        raise UnitError(message)
