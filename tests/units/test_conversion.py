import pytest

from bilt.units._conversion import Conversion, conversion_between
from bilt.units._units import kelvin, kilogram
from bilt.units.units import absolute_second, absolute_year, celsius, gram, year


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


def test_absolute_year_converts_through_a_non_adjacent_absolute_origin() -> None:
    """A year is defined through day/hour/minute, not directly through second."""
    seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0

    forward = conversion_between(absolute_year, absolute_second)
    backward = conversion_between(absolute_second, absolute_year)

    assert forward.scale == pytest.approx(seconds_per_year)
    assert forward.offset == 0.0
    assert backward.scale == pytest.approx(1.0 / seconds_per_year)
    assert backward.offset == 0.0
    assert year.to_si_base_units().relation.scale == pytest.approx(seconds_per_year)
