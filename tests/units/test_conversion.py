import pytest

from bilt.units._conversion import Conversion, conversion_between
from bilt.units._units import kelvin, kilogram
from bilt.units.units import celsius, gram


def test_same_absolute_unit_returns_identity_conversion() -> None:
    unit = abs(kelvin)

    assert conversion_between(unit, unit) == Conversion()


def test_relative_conversion_returns_scale() -> None:
    result = conversion_between(kilogram, gram)

    assert result.scale == pytest.approx(1_000.0)
    assert result.offset == 0.0


def test_absolute_conversion_returns_scale_and_offset() -> None:
    result = conversion_between(
        abs(celsius),
        abs(kelvin),
    )

    assert result.scale == pytest.approx(1.0)
    assert result.offset == pytest.approx(273.15)
