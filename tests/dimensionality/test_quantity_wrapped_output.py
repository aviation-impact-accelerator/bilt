"""An output port that carries a quantity answers with one, in coherent SI units, even when every input was a float."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.units import Quantity
from bilt.units.dimensions import DIMENSIONLESS, VOLTAGE
from bilt.units.units import volt

Voltage = typing.Annotated[Quantity, VOLTAGE]


def test_a_unit_bearing_output_given_only_floats_answers_in_its_inferred_dimensionality() -> None:
    @system
    class Rig(metaclass=Interface):
        x: float = field()
        doubled: Quantity = field()

    @equation
    def doubling(doubled: Quantity, x: Voltage) -> bool:
        return doubled == x * 2.0

    with Implementation(Rig, [doubling]):
        result = Rig(x=2.0).doubled

    assert result == 4.0 * volt


def test_a_unit_bearing_output_of_unresolved_dimensionality_answers_with_a_dimensionless_quantity() -> None:
    @system
    class Rig(metaclass=Interface):
        x: float = field()
        doubled: Quantity = field()

    @equation
    def doubling(doubled: Quantity, x: float) -> bool:
        return doubled == x * 2.0

    with Implementation(Rig, [doubling]):
        result = Rig(x=2.0).doubled

    assert isinstance(result, Quantity)
    assert result.value == pytest.approx(4.0)
    assert result.unit.dimension == DIMENSIONLESS
