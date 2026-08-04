import pytest

from bilt.units._conversion import convert_quantity_value
from bilt.units._quantities import Quantity
from bilt.units._units import kelvin, kilogram
from bilt.units.units import celsius, gram


def test_same_absolute_unit_returns_none() -> None:
    unit = abs(kelvin)

    assert convert_quantity_value(Quantity(273.15, unit), unit) is None


def test_relative_conversion_returns_scaled_value() -> None:
    result = convert_quantity_value(
        Quantity(1.0, kilogram),
        gram,
    )

    assert result == pytest.approx(1_000.0)


def test_absolute_conversion_applies_scale_and_offset() -> None:
    result = convert_quantity_value(
        Quantity(0.0, abs(celsius)),
        abs(kelvin),
    )

    assert result == pytest.approx(273.15)
