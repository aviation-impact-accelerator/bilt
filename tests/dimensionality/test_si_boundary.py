"""Quantities reach a backend as bare numbers in coherent SI units, and the answer comes back in them."""

import typing

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, equation, field, system
from bilt.errors import BiltInputDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import AREA, FORCE, LENGTH, MASS, TIME
from bilt.units.units import gram, hour, kilogram, kilometer, meter, newton, second

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend

Acceleration = typing.Annotated[Quantity, LENGTH / TIME**2]
Area = typing.Annotated[Quantity, AREA]
Force = typing.Annotated[Quantity, FORCE]
Length = typing.Annotated[Quantity, LENGTH]
Mass = typing.Annotated[Quantity, MASS]


@system
class Route(metaclass=Interface):
    outbound: Length = field()
    inbound: Length = field()
    total: Length = field()


@equation
def journey(total: Length, outbound: Length, inbound: Length) -> bool:
    return total == outbound + inbound


@system
class Body(metaclass=Interface):
    f: Force = field()
    m: Mass = field()
    a: Acceleration = field()


@equation
def newtons_second_law(f: Force, m: Mass, a: Acceleration) -> bool:
    return f == m * a


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_kilometres_and_metres_add_in_metres(backend: Backend) -> None:
    with backend, Implementation(Route, [journey]):
        result = Route(outbound=5.0 * kilometer, inbound=300.0 * meter).total

    assert result == 5300.0 * meter


def test_a_product_of_non_si_quantities_comes_back_in_coherent_si_units() -> None:
    with Implementation(Body, [newtons_second_law]):
        result = Body(m=1000.0 * gram, a=2.0 * meter / second**2).f

    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(2.0)
    assert result.unit == newton


def test_a_rearranged_equation_answers_in_coherent_si_units() -> None:
    with Implementation(Body, [newtons_second_law]):
        result = Body(f=6.0 * newton, a=2.0 * kilometer / hour**2).m

    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(38880.0)
    assert result.unit == kilogram


def test_quantities_that_contradict_a_declared_port_raise_at_instantiation() -> None:
    @system
    class Plot(metaclass=Interface):
        width: Length = field()
        depth: Length = field()
        area: Area = field()

    with pytest.raises(BiltInputDimensionalityError, match="'width'"):
        Plot(width=2.0 * kilogram, depth=3.0 * meter)
