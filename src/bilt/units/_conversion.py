import typing

from bilt.units._category import UnitCategory
from bilt.units._errors import BiltDimensionalityError, BiltUnitError

if typing.TYPE_CHECKING:
    from bilt.units._units import Unit


class QuantityLike(typing.Protocol):
    @property
    def value(self) -> float: ...

    @property
    def unit(self) -> Unit: ...

    def __str__(self) -> str: ...


def _create_absolute_unit_stack(unit: Unit) -> list[Unit]:
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
        raise BiltUnitError(message)

    return _create_absolute_unit_stack(unit.offset.composition[0].unit) + stack


def _validate_matching_dimensions(source: Unit, target: Unit) -> None:
    if source.dimension == target.dimension:
        return

    message = (
        f"Cannot convert quantity with unit dimensionality {source.dimension!s} "
        f"to unit {target!s} with dimensionality {target.dimension!s} as they "
        f"are different"
    )
    raise BiltDimensionalityError(message)


def _find_common_unit_index(
    start_unit_stack: list[Unit],
    end_unit_stack: list[Unit],
) -> int:
    index = 0

    for start_unit, end_unit in zip(  # noqa: B905
        start_unit_stack,
        end_unit_stack,
    ):
        if start_unit != end_unit:
            break
        index += 1

    return index


def _convert_absolute_quantity_value(
    quantity: QuantityLike,
    unit: Unit,
) -> float | None:
    if quantity.unit is unit:
        return None

    _validate_matching_dimensions(quantity.unit, unit)

    start_unit_stack = _create_absolute_unit_stack(quantity.unit)
    end_unit_stack = _create_absolute_unit_stack(unit)

    if start_unit_stack[0] is not end_unit_stack[0]:
        message = (
            f"Cannot convert quantity with unit {quantity!s} to unit {unit!s} "
            f"as the two do not share a common root unit in their definition"
        )
        raise BiltUnitError(message)

    index = _find_common_unit_index(
        start_unit_stack,
        end_unit_stack,
    )

    value = quantity.value

    for via_unit in reversed(start_unit_stack[index:]):
        scale = via_unit.relation.composition[0].unit.relation.scale
        value *= scale

        if via_unit.offset is not None:
            value += via_unit.offset.scale

    for via_unit in end_unit_stack[index:]:
        if via_unit.offset is not None:
            value -= via_unit.offset.scale

        scale = via_unit.relation.composition[0].unit.relation.scale
        value /= scale

    return value


def _convert_relative_quantity_value(
    quantity: QuantityLike,
    unit: Unit,
) -> float:
    _validate_matching_dimensions(quantity.unit, unit)

    if quantity.unit.metadata is not unit.metadata:
        message = (
            f"Cannot convert quantity with unit with metadata "
            f"{quantity.unit.metadata!s} to unit {unit!s} with metadata "
            f"{unit.metadata!s} as they are different"
        )
        raise BiltUnitError(message)

    scale = quantity.unit.to_si_base_units().relation.scale / unit.to_si_base_units().relation.scale

    return scale * quantity.value


def _raise_relative_to_absolute_error(
    quantity: QuantityLike,
    unit: Unit,
) -> typing.Never:
    message = (
        f"Cannot convert quantity with unit {quantity!s} to quantity with "
        f"absolute unit {unit} because relative and absolute units cannot be "
        f"interconverted"
    )
    raise BiltUnitError(message)


def _raise_absolute_to_relative_error(
    quantity: QuantityLike,
    unit: Unit,
) -> typing.Never:
    message = (
        f"Cannot convert quantity with absolute unit {quantity!s} to quantity "
        f"with unit {unit} because relative and absolute units cannot be "
        f"interconverted"
    )
    raise BiltUnitError(message)


def convert_quantity_value(
    quantity: QuantityLike,
    unit: Unit,
) -> float | None:
    """Return the converted value.

    ``None`` preserves the existing behaviour where converting an absolute
    quantity to the identical unit returns the original ``Quantity`` object.
    """
    source_category = quantity.unit.category
    target_category = unit.category

    if source_category is UnitCategory.ABSOLUTE and target_category is UnitCategory.ABSOLUTE:
        return _convert_absolute_quantity_value(quantity, unit)

    if target_category is UnitCategory.ABSOLUTE:
        _raise_relative_to_absolute_error(quantity, unit)

    if source_category is UnitCategory.ABSOLUTE:
        _raise_absolute_to_relative_error(quantity, unit)

    return _convert_relative_quantity_value(quantity, unit)
