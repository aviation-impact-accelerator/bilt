import typing

import pytest

from bilt import equation
from bilt.errors import BiltDimensionalityError, BiltEquationDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import AREA, DIMENSIONLESS, ENERGY, FORCE, LENGTH, MASS, PRESSURE, TIME

Area = typing.Annotated[Quantity, AREA]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Energy = typing.Annotated[Quantity, ENERGY]
Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]
Pressure = typing.Annotated[Quantity, PRESSURE]
Time = typing.Annotated[Quantity, TIME]
Velocity = typing.Annotated[Quantity, LENGTH / TIME]


def test_homogeneous_quotient_equation_decorates() -> None:
    @equation
    def pressure(p: Pressure, f: Force, area: Area) -> bool:
        return p == f / area


def test_equation_with_literal_factor_decorates() -> None:
    @equation
    def kinetic_energy(e: Energy, m: Mass, v: Velocity) -> bool:
        return e == 0.5 * m * v**2


def test_difference_in_a_denominator_decorates() -> None:
    @equation
    def speed(v: Velocity, d: Length, t1: Time, t0: Time) -> bool:
        return v == d / (t1 - t0)


def test_balance_against_zero_decorates() -> None:
    @equation
    def balance(inflow: Mass, outflow: Mass) -> bool:
        return inflow - outflow == 0


def test_inhomogeneous_sides_raise() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="pressure_vs_force") as raised:

        @equation
        def pressure_vs_force(p: Pressure, f: Force, d: Length) -> bool:
            return p == f * d

    message = str(raised.value)
    assert str(ENERGY) in message
    assert str(PRESSURE) in message


def test_inhomogeneous_addition_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError) as raised:

        @equation
        def summation(total: Mass, a: Mass, b: Length, c: Mass) -> bool:
            return total == a + b + c

    assert isinstance(raised.value, BiltDimensionalityError)
    assert isinstance(raised.value.__cause__, BiltDimensionalityError)


def test_unannotated_addend_does_not_hide_inhomogeneity() -> None:
    with pytest.raises(BiltEquationDimensionalityError):

        @equation
        def summation(total: float, a: Mass, b: Quantity, c: Length) -> bool:
            return total == a + b + c


def test_unannotated_factor_leaves_its_product_unconstrained() -> None:
    @equation
    def scaled(total: Force, m: Mass, k: float) -> bool:
        return total == m * k


def test_dimensional_exponent_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="dimensional_exponent"):

        @equation
        def dimensional_exponent(y: Length, x: Length, n: Dimensionless) -> bool:
            return y == x**n


def test_a_power_no_unannotated_port_can_meet_raises() -> None:
    with pytest.raises(BiltEquationDimensionalityError, match="admits no dimensionality for 'b'"):

        @equation
        def squared(a: Length, b: Quantity) -> bool:
            return a == b**2


def test_fully_unannotated_equation_decorates() -> None:
    @equation
    def summation(a: float, b: float, c: float) -> bool:
        return a == b + c
