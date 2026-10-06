"""Dimensionality agreed across an implementation's constituents by shared port name."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.errors import BiltImplementationDimensionalityError, BiltUnitsError
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, FORCE, MASS, POWER, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, volt, watt

Current = typing.Annotated[Quantity, CURRENT]
Force = typing.Annotated[Quantity, FORCE]
Mass = typing.Annotated[Quantity, MASS]
Power = typing.Annotated[Quantity, POWER]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]


@equation
def ohms_law(v: Voltage, i: Current, r: Resistance) -> bool:
    return v == i * r


@equation
def doubling(scaled: Voltage, v: Voltage) -> bool:
    return scaled == v * 2.0


@equation
def halving(halved: Voltage, scaled: Voltage) -> bool:
    return halved == scaled / 2.0


def test_constituents_agreeing_on_a_shared_port_construct() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()
        scaled: Voltage = field()

    Implementation(Rig, [ohms_law, doubling])


def test_a_port_only_one_constituent_declares_raises_naming_the_declaring_owner_and_the_plain_one() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()
        scaled: Voltage = field()

    @equation
    def plain_doubling(scaled: float, v: float) -> bool:
        return scaled == v * 2.0

    with pytest.raises(
        BiltImplementationDimensionalityError, match="'scaled' a dimensionality in only some places"
    ) as raised:
        Implementation(Rig, [ohms_law, plain_doubling])

    message = str(raised.value)
    assert f"{VOLTAGE}: declared by Rig" in message
    assert "left plain by plain_doubling" in message


def test_a_declared_conflict_names_both_sources() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()

    @equation
    def resting(v: Mass) -> bool:
        return v == 0

    with pytest.raises(BiltImplementationDimensionalityError, match="'v' conflicting dimensionalities") as raised:
        Implementation(Rig, [ohms_law, resting])

    message = str(raised.value)
    assert f"{VOLTAGE}: declared by Rig" in message
    assert f"{MASS}: declared by resting" in message


def test_a_conflict_reads_the_same_whatever_the_constituent_order() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()
        scaled: Force = field()
        halved: Voltage = field()

    messages = set()
    for constituents in ([ohms_law, doubling, halving], [halving, doubling, ohms_law]):
        with pytest.raises(BiltImplementationDimensionalityError) as raised:
            Implementation(Rig, constituents)
        messages.add(str(raised.value))

    assert len(messages) == 1


def test_a_sub_system_holds_its_ports_to_their_declarations_under_their_dotted_names() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        r: Resistance = field()

    with pytest.raises(BiltUnitsError, match="wiring") as raised:

        @equation
        def wiring(out: Mass, resistor: Resistor) -> bool:
            return out == resistor.v

    message = str(raised.value)
    assert "mass" in message
    assert "voltage" in message


def test_a_transform_output_takes_its_return_annotation() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        power: Mass = field()

    @transform
    def power(v: Voltage, i: Current) -> Power:
        return v * i

    with pytest.raises(BiltImplementationDimensionalityError, match="'power' conflicting dimensionalities") as raised:
        Implementation(Rig, [power])

    message = str(raised.value)
    assert f"{MASS}: declared by Rig" in message
    assert f"{POWER}: declared by power" in message


def test_a_transform_output_answers_in_the_dimensionality_of_its_return_annotation() -> None:
    @system
    class Rig(metaclass=Interface):
        v: Voltage = field()
        i: Current = field()
        power: Power = field()

    @transform
    def power(v: Voltage, i: Current) -> Power:
        return v * i

    with Implementation(Rig, [power]):
        result = Rig(v=6.0 * volt, i=2.0 * ampere).power

    assert result == 12.0 * watt
