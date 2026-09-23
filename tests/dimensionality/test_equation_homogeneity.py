"""Dimensional analysis for `@equation`, performed over the symbolic tree at decoration."""

import typing

import pytest

from bilt import equation
from bilt.errors import BiltUnitsError
from bilt.units import Quantity
from bilt.units.dimensions import AREA, FORCE, LENGTH, MASS, TIME, VOLTAGE

Area = typing.Annotated[Quantity, AREA]
Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Voltage = typing.Annotated[Quantity, VOLTAGE]
Acceleration = typing.Annotated[Quantity, LENGTH / TIME**2]


def test_homogeneous_equation_decorates() -> None:
    @equation
    def newtons_second_law(f: Force, m: Mass, a: Acceleration) -> bool:
        return f == m * a


def test_partially_annotated_equation_decorates() -> None:
    @equation
    def newtons_second_law(f: Force, m: Mass, a: float) -> bool:
        return f == m * a


def test_integer_exponent_decorates() -> None:
    @equation
    def square(area: Area, side: Length) -> bool:
        return area == side**2


@pytest.mark.xfail(reason="equation sides are not compared for dimensional homogeneity yet")
def test_inhomogeneous_equation_raises() -> None:
    with pytest.raises(BiltUnitsError, match="mislabelled") as raised:

        @equation
        def mislabelled(v: Voltage, m: Mass, a: Acceleration) -> bool:
            return v == m * a

    message = str(raised.value)
    assert "voltage" in message
    assert "force" in message


@pytest.mark.xfail(reason="addition of unequal dimensionalities is not rejected yet")
def test_adding_unequal_dimensionalities_raises() -> None:
    with pytest.raises(BiltUnitsError, match="summation") as raised:

        @equation
        def summation(total: Mass, a: Mass, b: Length) -> bool:
            return total == a + b

    message = str(raised.value)
    assert "mass" in message
    assert "length" in message


@pytest.mark.xfail(reason="a dimensional exponent is not rejected yet")
def test_dimensional_exponent_raises() -> None:
    with pytest.raises(BiltUnitsError, match="dimensional_exponent") as raised:

        @equation
        def dimensional_exponent(result: Length, base: Length, m: Mass) -> bool:
            return result == base**m

    assert "mass" in str(raised.value)
