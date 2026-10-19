import dataclasses
import typing

from bilt.units._category import UnitCategory
from bilt.units._errors import BiltDimensionalityError, BiltUnitError

if typing.TYPE_CHECKING:
    from bilt.units._units import Unit


@dataclasses.dataclass(frozen=True, slots=True)
class Conversion:
    scale: float = 1.0
    offset: float = 0.0

    def apply(self, value: float) -> float:
        return value * self.scale + self.offset


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
        f"Cannot convert unit with dimensionality {source.dimension!s} "
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


def _absolute_conversion(
    source: Unit,
    target: Unit,
) -> Conversion:
    _validate_matching_dimensions(source, target)

    start_unit_stack = _create_absolute_unit_stack(source)
    end_unit_stack = _create_absolute_unit_stack(target)

    if start_unit_stack[0] is not end_unit_stack[0]:
        message = (
            f"Cannot convert unit {source!s} to unit {target!s} "
            f"as the two do not share a common root unit in their definition"
        )
        raise BiltUnitError(message)

    index = _find_common_unit_index(
        start_unit_stack,
        end_unit_stack,
    )

    scale = 1.0
    offset = 0.0

    for via_unit in reversed(start_unit_stack[index:]):
        via_scale = via_unit.relation.composition[0].unit.relation.scale

        scale *= via_scale
        offset *= via_scale

        if via_unit.offset is not None:
            offset += via_unit.offset.scale

    for via_unit in end_unit_stack[index:]:
        if via_unit.offset is not None:
            offset -= via_unit.offset.scale

        via_scale = via_unit.relation.composition[0].unit.relation.scale

        scale /= via_scale
        offset /= via_scale

    return Conversion(scale=scale, offset=offset)


def _relative_conversion(
    source: Unit,
    target: Unit,
) -> Conversion:
    _validate_matching_dimensions(source, target)

    scale = source.to_si_base_units().relation.scale / target.to_si_base_units().relation.scale

    return Conversion(scale=scale)


def _raise_relative_to_absolute_error(
    source: Unit,
    target: Unit,
) -> typing.Never:
    message = (
        f"Cannot convert relative unit {source!s} to absolute unit {target!s} "
        f"because relative and absolute units cannot be interconverted"
    )
    raise BiltUnitError(message)


def _raise_absolute_to_relative_error(
    source: Unit,
    target: Unit,
) -> typing.Never:
    message = (
        f"Cannot convert absolute unit {source!s} to relative unit {target!s} "
        f"because relative and absolute units cannot be interconverted"
    )
    raise BiltUnitError(message)


def conversion_between(
    source: Unit,
    target: Unit,
) -> Conversion:
    source_category = source.category
    target_category = target.category

    if source_category is UnitCategory.ABSOLUTE and target_category is UnitCategory.ABSOLUTE:
        return _absolute_conversion(source, target)

    if target_category is UnitCategory.ABSOLUTE:
        _raise_relative_to_absolute_error(source, target)

    if source_category is UnitCategory.ABSOLUTE:
        _raise_absolute_to_relative_error(source, target)

    return _relative_conversion(source, target)
