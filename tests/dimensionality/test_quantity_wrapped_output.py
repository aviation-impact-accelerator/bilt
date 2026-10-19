"""A declared output port answers with a quantity in the coherent SI unit of its declared dimensionality."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, DIMENSIONLESS, MASS, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, kilogram, millivolt, ohm, volt

Current = typing.Annotated[Quantity, CURRENT]
Dimensionless = typing.Annotated[Quantity, DIMENSIONLESS]
Mass = typing.Annotated[Quantity, MASS]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]


@system
class Resistor(metaclass=Interface):
    v: Voltage = field()
    i: Current = field()
    r: Resistance = field()


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


def test_a_declared_output_answers_in_the_coherent_si_unit_of_its_dimensionality() -> None:
    with Implementation(Resistor, [ohms_law]):
        result = Resistor(i=2.0 * ampere, r=3000.0 * millivolt / ampere).v

    assert isinstance(result, Quantity)
    assert result.unit == volt
    assert result.value == pytest.approx(6.0)


def test_a_declared_output_takes_its_dimensionality_from_its_own_declaration() -> None:
    with Implementation(Resistor, [ohms_law]):
        result = Resistor(v=6000.0 * millivolt, i=2.0 * ampere).r

    assert result.unit == ohm
    assert result.value == pytest.approx(3.0)


def test_a_ratio_answers_as_a_dimensionless_quantity_rather_than_a_bare_number() -> None:
    @system
    class Split(metaclass=Interface):
        part: Mass = field()
        whole: Mass = field()
        fraction: Dimensionless = field()

    @equation
    def splitting(fraction: Dimensionless, part: Mass, whole: Mass) -> bool:
        return fraction == part / whole

    with Implementation(Split, [splitting]):
        result = Split(part=2.0 * kilogram, whole=8.0 * kilogram).fraction

    assert isinstance(result, Quantity)
    assert result.unit.dimension == DIMENSIONLESS
    assert result.value == pytest.approx(0.25)
