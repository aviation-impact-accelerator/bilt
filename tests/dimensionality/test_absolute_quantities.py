"""An absolute port holds a quantity measured from a fixed zero, and answers in the absolute coherent unit."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.errors import BiltDefinitionError, BiltEquationDimensionalityError, BiltInputDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import ABSOLUTE, AMOUNT, DIMENSIONLESS, ENERGY, MASS, PRESSURE, TEMPERATURE, TIME, VOLUME
from bilt.units.units import (
    absolute_celsius,
    absolute_fahrenheit,
    absolute_kelvin,
    absolute_year,
    celsius,
    fahrenheit,
    joule,
    kelvin,
    kilogram,
    mole,
)

AbsoluteTemperature = typing.Annotated[Quantity, TEMPERATURE, ABSOLUTE]
Temperature = typing.Annotated[Quantity, TEMPERATURE]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]


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


def test_two_absolute_quantities_do_not_add() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match=r"'film_temperature' adds two absolute quantities"):

        @equation
        def film_temperature(film: AbsoluteTemperature, hot: AbsoluteTemperature, cold: AbsoluteTemperature) -> bool:
            return film == (hot + cold) / 2


def test_a_mean_of_absolute_quantities_is_written_from_one_of_them() -> None:
    @system
    class Wall(metaclass=Interface):
        hot: AbsoluteTemperature = field()
        cold: AbsoluteTemperature = field()
        film: AbsoluteTemperature = field()

    @equation
    def film_temperature(film: AbsoluteTemperature, hot: AbsoluteTemperature, cold: AbsoluteTemperature) -> bool:
        return film == cold + (hot - cold) / 2

    with Implementation(Wall, [film_temperature]):
        result = Wall(hot=Quantity(40.0, absolute_celsius), cold=Quantity(20.0, absolute_celsius)).film

    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(303.15)


def test_an_absolute_quantity_does_not_subtract_from_a_relative_one() -> None:
    with pytest.raises(
        BiltEquationDimensionalityError,
        match=r"'shortfall' subtracts an absolute quantity from a relative one",
    ):

        @equation
        def shortfall(excess: Temperature, rise: Temperature, inlet: AbsoluteTemperature) -> bool:
            return excess == rise - inlet


def test_a_product_is_neither_absolute_nor_relative() -> None:
    @equation
    def ideal_gas(
        pressure: typing.Annotated[Quantity, PRESSURE],
        volume: typing.Annotated[Quantity, VOLUME],
        amount: typing.Annotated[Quantity, AMOUNT],
        temperature: AbsoluteTemperature,
    ) -> bool:
        return pressure * volume == amount * (8.314 * joule / (mole * kelvin)) * temperature

    @equation
    def compression(outlet: AbsoluteTemperature, inlet: AbsoluteTemperature, ratio: Dimensionless) -> bool:
        return outlet == inlet * ratio

    @equation
    def carnot(efficiency: Dimensionless, cold: AbsoluteTemperature, hot: AbsoluteTemperature) -> bool:
        return efficiency == 1 - cold / hot  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]

    @equation
    def compression_rise(rise: Temperature, inlet: AbsoluteTemperature, ratio: Dimensionless) -> bool:
        return rise == inlet * (ratio - 1)  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]


def test_a_sum_holding_a_product_is_read_the_one_legal_way() -> None:
    with pytest.raises(
        BiltEquationDimensionalityError, match=r"'heated' relates an absolute quantity to a relative one"
    ):

        @equation
        def heated(
            rise: Temperature,
            inlet: AbsoluteTemperature,
            heat: typing.Annotated[Quantity, ENERGY],
            capacity: typing.Annotated[Quantity, ENERGY / TEMPERATURE],
        ) -> bool:
            return rise == inlet + heat / capacity


def test_a_relative_quantity_constant_related_to_an_absolute_port_raises() -> None:
    with pytest.raises(
        BiltEquationDimensionalityError, match=r"'thawed' relates an absolute quantity to a relative one"
    ):

        @equation
        def thawed(reading: AbsoluteTemperature) -> bool:
            return reading > 273.15 * kelvin


def test_an_absolute_quantity_constant_relates_to_an_absolute_port() -> None:
    @equation
    def thawed(reading: AbsoluteTemperature) -> bool:
        return reading > Quantity(0.0, absolute_celsius)


def test_a_sub_system_port_keeps_its_absoluteness_in_a_parent_equation() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match=r"'doubled' adds two absolute quantities"):

        @equation
        def doubled(total: AbsoluteTemperature, heater: Heater) -> bool:
            return total == heater.inlet + heater.outlet


@system
class Probe(metaclass=Interface):
    degrees: float = field()
    reading: AbsoluteTemperature = field()


def test_a_constructed_absolute_quantity_brings_a_plain_port_to_an_absolute_one() -> None:
    @transform
    def reading(degrees: float) -> AbsoluteTemperature:
        return Quantity(degrees, absolute_fahrenheit)

    with Implementation(Probe, [reading]):
        result = Probe(degrees=68.0).reading

    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(293.15)


def test_a_converted_value_takes_an_absolute_port_out_to_a_plain_one() -> None:
    @transform
    def degrees(reading: AbsoluteTemperature) -> float:
        return reading.convert_to(absolute_fahrenheit).value

    with Implementation(Probe, [degrees]):
        assert Probe(reading=Quantity(20.0, absolute_celsius)).degrees == pytest.approx(68.0)


def test_a_conversion_to_an_absolute_unit_answers_the_same_quantity() -> None:
    @transform
    def outlet(inlet: AbsoluteTemperature) -> AbsoluteTemperature:
        return inlet.convert_to(absolute_celsius)

    with Implementation(Heater, [outlet]):
        result = Heater(inlet=Quantity(20.0, absolute_celsius)).outlet

    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(293.15)


def test_a_body_working_in_degrees_means_what_python_means() -> None:
    @transform
    def outlet(inlet: AbsoluteTemperature) -> AbsoluteTemperature:
        return Quantity(inlet.convert_to(absolute_celsius).value + 10.0, absolute_celsius)

    with Implementation(Heater, [outlet]):
        result = Heater(inlet=Quantity(68.0, absolute_fahrenheit)).outlet

    assert result.unit is absolute_kelvin
    assert result.value == pytest.approx(303.15)


def test_converting_an_absolute_parameter_to_a_relative_unit_is_refused() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`inlet\.convert_to\(celsius\)` contradicts the signature: `inlet` is a Quantity of absolute "
        r"temperature.*converts only to an absolute unit such as \|kelvin\|",
    ):

        @transform
        def degrees(inlet: AbsoluteTemperature) -> float:
            return inlet.convert_to(celsius).value


def test_converting_a_relative_parameter_to_an_absolute_unit_is_refused() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`rise\.convert_to\(absolute_celsius\)` contradicts the signature: `rise` is a Quantity of "
        r"temperature.*converts only to a relative unit such as kelvin",
    ):

        @transform
        def outlet(rise: Temperature) -> AbsoluteTemperature:
            return rise.convert_to(absolute_celsius)


def test_a_conversion_of_anything_deeper_than_a_parameter_is_taken_at_its_word() -> None:
    @transform
    def degrees(reading: AbsoluteTemperature) -> float:
        return (reading - Quantity(0.0, absolute_celsius)).convert_to(fahrenheit).value

    with Implementation(Probe, [degrees]):
        assert Probe(reading=Quantity(10.0, absolute_celsius)).degrees == pytest.approx(18.0)


def test_a_conversion_of_a_reassigned_parameter_is_taken_at_its_word() -> None:
    @transform
    def degrees(reading: AbsoluteTemperature) -> float:
        reading = reading - Quantity(0.0, absolute_celsius)
        return reading.convert_to(fahrenheit).value

    with Implementation(Probe, [degrees]):
        assert Probe(reading=Quantity(10.0, absolute_celsius)).degrees == pytest.approx(18.0)


def test_a_comprehension_variable_named_as_a_parameter_does_not_reassign_it() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`reading\.convert_to\(celsius\)` contradicts the signature"):

        @transform
        def degrees(reading: AbsoluteTemperature) -> float:
            doubled = [reading * 2.0 for reading in [1.0, 2.0]]
            return reading.convert_to(celsius).value + doubled[0]


def test_a_conversion_in_the_iterable_of_a_comprehension_reads_the_parameter() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`reading\.convert_to\(celsius\)` contradicts the signature"):

        @transform
        def degrees(reading: AbsoluteTemperature) -> float:
            values = [reading for reading in [reading.convert_to(celsius).value] if reading > 0.0]
            return values[0]


def test_a_conversion_in_a_parameter_default_reads_the_parameter() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`reading\.convert_to\(celsius\)` contradicts the signature"):

        @transform
        def degrees(reading: AbsoluteTemperature) -> float:
            def scaled(reading: float = reading.convert_to(celsius).value) -> float:
                return reading

            return scaled()


def test_a_conversion_in_the_iterable_of_a_nested_comprehension_reads_the_parameter() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`reading\.convert_to\(celsius\)` contradicts the signature"):

        @transform
        def degrees(reading: AbsoluteTemperature) -> float:
            rows = [[reading for reading in [reading.convert_to(celsius).value] if reading > 0.0] for _ in [1.0]]
            return rows[0][0]


def test_a_binding_deeper_in_a_comprehension_does_not_shadow_the_parameter() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`reading\.convert_to\(celsius\)` contradicts the signature"):

        @transform
        def degrees(reading: AbsoluteTemperature) -> float:
            values = [reading.convert_to(celsius).value + sum(reading for reading in [1.0]) for _ in [1.0]]
            return values[0]


def test_a_conversion_in_a_parameter_annotation_reads_the_parameter() -> None:
    with pytest.raises(BiltDefinitionError, match=r"`reading\.convert_to\(celsius\)` contradicts the signature"):

        @transform
        def degrees(reading: AbsoluteTemperature) -> float:
            def scaled(reading: reading.convert_to(celsius).value = 1.0) -> float:
                return reading

            return scaled() + Quantity(0.0, celsius).value


def test_a_coherent_conversion_in_the_iterable_of_a_comprehension_is_still_erased() -> None:
    @transform
    def degrees(reading: AbsoluteTemperature) -> float:
        values = [reading for reading in [reading.convert_to(absolute_fahrenheit).value] if reading > 0.0]
        return values[0]

    with Implementation(Probe, [degrees]):
        assert Probe(reading=Quantity(10.0, absolute_celsius)).degrees == pytest.approx(50.0)


def test_an_absolute_unit_with_no_coherent_counterpart_is_refused_in_a_transform() -> None:
    with pytest.raises(
        BiltDefinitionError,
        match=r"`absolute_year` cannot be converted to a coherent unit, so `Quantity\(elapsed, absolute_year\)` "
        r"cannot use it",
    ):

        @transform
        def started(elapsed: float) -> typing.Annotated[Quantity, TIME]:
            return Quantity(elapsed, absolute_year)
