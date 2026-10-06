"""A parent and its sub-systems agree on the dimensionality of each port they share, at every level of nesting."""

import typing

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.errors import BiltImplementationDimensionalityError, BiltPortDeclarationError, BiltUnitsError
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, LENGTH, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, ohm, volt

Current = typing.Annotated[Quantity, CURRENT]
Length = typing.Annotated[Quantity, LENGTH]
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


@system
class Component(metaclass=Interface):
    voltage: Voltage = field()
    current: Current = field()
    resistance: Resistance = field()


@equation
def voltage_eq(voltage: Voltage, resistor: Resistor) -> bool:
    return voltage == resistor.v


@equation
def current_eq(current: Current, resistor: Resistor) -> bool:
    return current == resistor.i


@equation
def resistance_eq(resistance: Resistance, resistor: Resistor) -> bool:
    return resistance == resistor.r


def test_a_childs_declared_port_reaches_the_parent_under_its_dotted_name() -> None:
    resistor = Implementation(Resistor, [ohms_law])
    with Implementation(
        Component, [voltage_eq, current_eq, resistance_eq], internal_implementations={"resistor": resistor}
    ):
        result = Component(voltage=6.0 * volt, current=2.0 * ampere).resistance

    assert result == 3.0 * ohm
    assert result.unit.dimension == RESISTANCE


def test_a_parent_port_contradicting_what_the_child_declares_raises_at_decoration() -> None:
    with pytest.raises(BiltUnitsError, match="relates"):

        @equation
        def contradicting(resistance: Length, resistor: Resistor) -> bool:
            return resistance == resistor.r


def test_a_dimensionality_a_grandchild_declares_reaches_the_grandparent() -> None:
    @system
    class Circuit(metaclass=Interface):
        total: Length = field()

    with pytest.raises(BiltUnitsError, match=r"total_eq.*relates"):

        @equation
        def total_eq(total: Length, component: Component) -> bool:
            return total == component.resistance


def test_a_parent_constituent_naming_a_childs_declared_port_must_declare_its_own_ports() -> None:
    with pytest.raises(BiltPortDeclarationError, match="mixes declared and plain ports") as raised:

        @equation
        def link(voltage: float, resistor: Resistor) -> bool:
            return voltage == resistor.v

    message = str(raised.value)
    assert "declared: 'resistor.v'" in message
    assert "plain: 'voltage'" in message


def test_a_child_leaving_a_port_plain_where_the_parent_declares_it_raises_at_construction() -> None:
    @system
    class PlainResistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Parent(metaclass=Interface):
        voltage: Voltage = field()

    @equation
    def plain_voltage_eq(voltage: float, resistor: PlainResistor) -> bool:
        return voltage == resistor.v

    resistor = Implementation(PlainResistor, [])
    with pytest.raises(
        BiltImplementationDimensionalityError, match="'voltage' a dimensionality in only some places"
    ) as raised:
        Implementation(Parent, [plain_voltage_eq], internal_implementations={"resistor": resistor})

    assert "left plain by plain_voltage_eq" in str(raised.value)
