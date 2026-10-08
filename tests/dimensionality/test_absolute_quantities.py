"""An absolute port holds a quantity measured from a fixed zero, and answers in the absolute coherent unit."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.errors import BiltInputDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import ABSOLUTE, MASS, TEMPERATURE, TIME
from bilt.units.units import (
    absolute_celsius,
    absolute_fahrenheit,
    absolute_kelvin,
    absolute_year,
    celsius,
    kelvin,
    kilogram,
)

AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
Temperature = typing.Annotated[Quantity, TEMPERATURE]


@system
class Heater(metaclass=Interface):
    inlet: AbsoluteTemperature = field()
    rise: Temperature = field()
    outlet: AbsoluteTemperature = field()


@equation
def heating(outlet: AbsoluteTemperature, inlet: AbsoluteTemperature, rise: Temperature) -> bool:
    return outlet == inlet + rise


def test_an_absolute_port_counts_from_absolute_zero_and_answers_in_the_absolute_coherent_unit() -> None:
    with Implementation(Heater, [heating]):
        result = Heater(inlet=Quantity(20.0, absolute_celsius), rise=10.0 * celsius).outlet

    assert isinstance(result, Quantity)
    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(303.15)


def test_a_relative_port_solved_from_absolute_ports_answers_in_the_relative_coherent_unit() -> None:
    with Implementation(Heater, [heating]):
        result = Heater(inlet=Quantity(20.0, absolute_celsius), outlet=Quantity(86.0, absolute_fahrenheit)).rise

    assert isinstance(result, Quantity)
    assert result.unit is kelvin
    assert result.value == pytest.approx(10.0)


def test_a_relative_quantity_at_an_absolute_port_raises() -> None:
    with pytest.raises(
        BiltInputDimensionalityError, match="'inlet' of 'Heater' must be a Quantity of absolute temperature"
    ) as raised:
        Heater(inlet=20.0 * celsius, rise=10.0 * kelvin)

    message = str(raised.value)
    assert "in an absolute unit such as |kelvin|" in message
    assert "not 20.0 celsius" in message


def test_an_absolute_quantity_at_a_relative_port_raises() -> None:
    with pytest.raises(
        BiltInputDimensionalityError, match="'rise' of 'Heater' must be a Quantity of temperature"
    ) as raised:
        Heater(inlet=Quantity(20.0, absolute_celsius), rise=Quantity(10.0, absolute_celsius))

    message = str(raised.value)
    assert "in a relative unit such as kelvin" in message
    assert "not 10.0 |celsius|" in message


def test_a_bare_float_at_an_absolute_port_raises() -> None:
    with pytest.raises(
        BiltInputDimensionalityError, match="'inlet' of 'Heater' must be a Quantity of absolute temperature"
    ):
        Heater(inlet=20.0, rise=10.0 * kelvin)  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]


def test_an_absolute_unit_whose_dimensionality_cannot_be_absolute_raises() -> None:
    @system
    class Clock(metaclass=Interface):
        started: typing.Annotated[Quantity, TIME] = field()

    with pytest.raises(BiltInputDimensionalityError, match=r"'started' of 'Clock'.*in a relative unit such as second"):
        Clock(started=Quantity(2020.0, absolute_year))


def test_a_compound_unit_containing_an_absolute_unit_raises_at_a_relative_port() -> None:
    @system
    class Tank(metaclass=Interface):
        thermal_mass: typing.Annotated[Quantity, MASS * TEMPERATURE] = field()

    with pytest.raises(BiltInputDimensionalityError, match=r"'thermal_mass' of 'Tank'.*in a relative unit") as raised:
        Tank(thermal_mass=60.0 * kilogram * absolute_celsius)

    assert "not 60.0 kilogram |celsius|" in str(raised.value)


def test_a_compound_unit_containing_an_absolute_unit_raises_at_an_absolute_port() -> None:
    with pytest.raises(BiltInputDimensionalityError, match=r"'inlet' of 'Heater'.*in an absolute unit"):
        Heater(inlet=Quantity(2.0, absolute_celsius**2 / celsius), rise=10.0 * kelvin)


def test_an_absolute_quantity_constant_in_an_equation_counts_from_absolute_zero() -> None:
    @system
    class Thermometer(metaclass=Interface):
        above_freezing: Temperature = field()
        reading: AbsoluteTemperature = field()

    @equation
    def reading_from_freezing(reading: AbsoluteTemperature, above_freezing: Temperature) -> bool:
        return reading == Quantity(0.0, absolute_celsius) + above_freezing

    with Implementation(Thermometer, [reading_from_freezing]):
        result = Thermometer(above_freezing=20.0 * kelvin).reading

    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(293.15)


def test_an_absolute_quantity_constant_in_a_transform_counts_from_absolute_zero() -> None:
    @system
    class Thermometer(metaclass=Interface):
        above_freezing: Temperature = field()
        reading: AbsoluteTemperature = field()

    @transform
    def reading(above_freezing: Temperature) -> AbsoluteTemperature:
        return Quantity(0.0, absolute_celsius) + above_freezing

    with Implementation(Thermometer, [reading]):
        result = Thermometer(above_freezing=20.0 * kelvin).reading

    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(293.15)
