import dataclasses
import typing

import pytest

from bilt.units._category import UnitCategory
from bilt.units._conversion import convert_quantity_value
from bilt.units._errors import UnitError
from bilt.units._quantities import Quantity
from bilt.units._units import Unit, kelvin, kilogram
from bilt.units.units import celsius, gram


def test_same_absolute_unit_returns_none() -> None:
    unit = abs(kelvin)
    quantity = Quantity(273.15, unit)

    assert convert_quantity_value(quantity, unit) is None


def test_relative_conversion_returns_scaled_value() -> None:
    quantity = Quantity(1.0, kilogram)

    result = convert_quantity_value(quantity, gram)

    assert result == pytest.approx(1_000.0)


def test_absolute_conversion_applies_scale_and_offset() -> None:
    quantity = Quantity(0.0, abs(celsius))

    result = convert_quantity_value(quantity, abs(kelvin))

    assert result == pytest.approx(273.15)


@dataclasses.dataclass(frozen=True)
class _Composition:
    unit: object


@dataclasses.dataclass(frozen=True)
class _Relation:
    composition: tuple[_Composition, ...]
    scale: float = 1.0


class _RelativeUnit:
    def __init__(self, *, is_si_base_unit: bool) -> None:
        self.is_si_base_unit = is_si_base_unit


class _AbsoluteUnit:
    category = UnitCategory.ABSOLUTE
    metadata = None

    def __init__(
        self,
        name: str,
        *,
        dimension: object,
        relative_unit: _RelativeUnit,
        offset: object | None = None,
    ) -> None:
        self.name = name
        self.dimension = dimension
        self.relation = _Relation(
            composition=(_Composition(relative_unit),),
        )
        self.offset = offset

    def __str__(self) -> str:
        return self.name


def test_absolute_units_without_common_root_raise() -> None:
    dimension = object()

    source = typing.cast(
        "Unit",
        _AbsoluteUnit(
            "source",
            dimension=dimension,
            relative_unit=_RelativeUnit(is_si_base_unit=True),
        ),
    )
    target = typing.cast(
        "Unit",
        _AbsoluteUnit(
            "target",
            dimension=dimension,
            relative_unit=_RelativeUnit(is_si_base_unit=True),
        ),
    )

    with pytest.raises(
        UnitError,
        match="do not share a common root unit",
    ):
        convert_quantity_value(
            Quantity(1.0, source),
            target,
        )


def test_absolute_unit_chain_without_offset_raises() -> None:
    dimension = object()

    source = typing.cast(
        "Unit",
        _AbsoluteUnit(
            "source",
            dimension=dimension,
            relative_unit=_RelativeUnit(is_si_base_unit=False),
            offset=None,
        ),
    )
    target = typing.cast(
        "Unit",
        _AbsoluteUnit(
            "target",
            dimension=dimension,
            relative_unit=_RelativeUnit(is_si_base_unit=True),
        ),
    )

    with pytest.raises(UnitError, match="has no offset"):
        convert_quantity_value(
            Quantity(1.0, source),
            target,
        )
