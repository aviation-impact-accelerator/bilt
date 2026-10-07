"""A quantity or unit written in an equation is a constant that carries its dimensionality and its coherent value."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.errors import BiltEquationDimensionalityError, BiltSymbolicConversionError
from bilt.units import UNITLESS, Quantity
from bilt.units.dimensions import DIMENSIONLESS, FORCE, LENGTH, MASS, TEMPERATURE, TIME
from bilt.units.units import absolute_celsius, hour, kilogram, kilometer, meter, newton, second

Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Temperature = typing.Annotated[Quantity, TEMPERATURE]
Time = typing.Annotated[Quantity, TIME]


@system
class Body(metaclass=Interface):
    force: Force = field()
    mass: Mass = field()


@equation
def weight(force: Force, mass: Mass) -> bool:
    return force == mass * (9.81 * meter / second**2)


def test_an_equation_with_a_quantity_constant_is_solved_for_its_left_side() -> None:
    with Implementation(Body, [weight]):
        result = Body(mass=2.0 * kilogram).force

    assert result.value == pytest.approx(19.62)
    assert result.unit == newton


def test_an_equation_with_a_quantity_constant_is_solved_for_its_right_side() -> None:
    with Implementation(Body, [weight]):
        result = Body(force=19.62 * newton).mass

    assert result.value == pytest.approx(2.0)
    assert result.unit == kilogram


def test_a_quantity_constant_written_before_its_operand_means_the_same() -> None:
    @equation
    def weight_reversed(force: Force, mass: Mass) -> bool:
        return force == (9.81 * meter / second**2) * mass

    with Implementation(Body, [weight_reversed]):
        result = Body(mass=2.0 * kilogram).force

    assert result.value == pytest.approx(19.62)
    assert result.unit == newton


def test_a_quantity_constant_referred_to_by_name_means_the_same() -> None:
    gravity = 9.81 * meter / second**2

    @equation
    def weight_by_name(force: Force, mass: Mass) -> bool:
        return force == mass * gravity

    with Implementation(Body, [weight_by_name]):
        result = Body(mass=2.0 * kilogram).force

    assert result.value == pytest.approx(19.62)
    assert result.unit == newton


def test_a_constant_in_another_unit_means_what_it_says() -> None:
    @system
    class Journey(metaclass=Interface):
        distance: Length = field()
        duration: Time = field()

    @equation
    def cruise(distance: Length, duration: Time) -> bool:
        return distance == duration * (36.0 * kilometer / hour)

    with Implementation(Journey, [cruise]):
        result = Journey(duration=2.0 * second).distance

    assert result.value == pytest.approx(20.0)
    assert result.unit == meter


def test_a_bare_unit_is_one_of_that_unit() -> None:
    @system
    class Route(metaclass=Interface):
        distance: Length = field()
        laps: Dimensionless = field()

    @equation
    def lapping(distance: Length, laps: Dimensionless) -> bool:
        return distance == laps * kilometer

    with Implementation(Route, [lapping]):
        result = Route(laps=Quantity(2.0, UNITLESS)).distance

    assert result.value == pytest.approx(2000.0)
    assert result.unit == meter


def test_a_constant_of_the_wrong_dimensionality_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="mislabelled"):

        @equation
        def mislabelled(force: Force, mass: Mass) -> bool:
            return force == mass * (9.81 * meter / second)


def test_a_zero_quantity_constant_keeps_its_own_dimensionality() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="relates mass"):

        @equation
        def grounded(mass: Mass) -> bool:
            return mass == 0.0 * meter


def test_an_absolute_unit_as_a_factor_raises() -> None:
    with pytest.raises(BiltSymbolicConversionError, match=r"absolute unit '\|celsius\|'.*relative unit 'celsius'"):

        @equation
        def reading(temperature: Temperature, count: Dimensionless) -> bool:
            return temperature == count * absolute_celsius


def test_a_plain_port_meeting_a_quantity_constant_raises() -> None:
    def scaled(y: float, x: float) -> bool:
        return y == x * (2.0 * meter)

    with pytest.raises(
        BiltEquationDimensionalityError,
        match=r"quantity constant in one operation: plain 'x' with a constant of length",
    ) as raised:
        equation(scaled)

    assert "declared" not in str(raised.value)


def test_a_plain_port_meeting_a_dimensionless_quantity_constant_raises() -> None:
    def doubled(y: float, x: float) -> bool:
        return y == x * Quantity(2.0, UNITLESS)

    with pytest.raises(BiltEquationDimensionalityError, match="quantity constant"):
        equation(doubled)
