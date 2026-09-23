"""Dimensionality unified across an implementation's constituents by shared port name."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.errors import BiltUnitsError
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, FORCE, MASS, RESISTANCE, VOLTAGE

Current = typing.Annotated[Quantity, CURRENT]
Force = typing.Annotated[Quantity, FORCE]
Mass = typing.Annotated[Quantity, MASS]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


def test_constituents_agreeing_on_a_shared_port_construct() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()
        scaled: Voltage = field()

    @equation
    def doubling(scaled: Voltage, v: Voltage) -> bool:
        return scaled == v * 2.0

    Implementation(Rig, [ohms_law, doubling])


def test_a_port_only_one_constituent_annotates_constructs() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()
        scaled: float = field()

    @equation
    def doubling(scaled: float, v: float) -> bool:
        return scaled == v * 2.0

    Implementation(Rig, [ohms_law, doubling])


@pytest.mark.xfail(reason="dimensionality is not unified across constituents at construction yet")
def test_constituents_disagreeing_on_a_shared_port_raise() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()
        scaled: Force = field()

    @equation
    def doubling(scaled: Force, v: Mass) -> bool:
        return scaled == v * 2.0

    with pytest.raises(BiltUnitsError, match="'v'") as raised:
        Implementation(Rig, [ohms_law, doubling])

    message = str(raised.value)
    assert "voltage" in message
    assert "mass" in message
    assert "ohms_law" in message
    assert "doubling" in message
