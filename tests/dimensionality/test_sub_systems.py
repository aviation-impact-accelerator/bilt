"""A parent and its sub-systems agree on the dimensionality of each port they share, at every level of nesting."""

import typing

import pytest

from bilt import PYTHON_BACKEND, Implementation, Interface, equation, field, system
from bilt.errors import BiltImplementationDimensionalityError, BiltInputDimensionalityError
from bilt.units import Quantity
from bilt.units.dimensions import CURRENT, LENGTH, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, meter, ohm, volt

Current = typing.Annotated[Quantity, CURRENT]
Length = typing.Annotated[Quantity, LENGTH]
Voltage = typing.Annotated[Quantity, VOLTAGE]


@system
class Resistor(metaclass=Interface):
    v: Voltage = field()
    i: Current = field()
    r: Quantity = field()


@system
class UnconstrainedResistor(metaclass=Interface):
    v: Quantity = field()
    i: Quantity = field()
    r: Quantity = field()


@equation
def ohms_law(v: Quantity, i: Quantity, r: Quantity) -> bool:
    return v == i * r


def test_a_dimensionality_the_child_infers_reaches_the_parent() -> None:
    @system
    class Component(metaclass=Interface):
        voltage: Quantity = field()
        current: Quantity = field()
        resistance: Quantity = field()

    @equation
    def voltage_eq(voltage: Quantity, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: Quantity, resistor: Resistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: Quantity, resistor: Resistor) -> bool:
        return resistance == resistor.r

    resistor = Implementation(Resistor, [ohms_law])
    with Implementation(
        Component, [voltage_eq, current_eq, resistance_eq], internal_implementations={"resistor": resistor}
    ):
        result = Component(voltage=6.0 * volt, current=2.0 * ampere).resistance

    assert result == 3.0 * ohm
    assert result.unit.dimension == RESISTANCE


def test_a_parent_port_contradicting_what_the_child_infers_raises() -> None:
    @system
    class Component(metaclass=Interface):
        resistance: Length = field()

    @equation
    def resistance_eq(resistance: Quantity, resistor: Resistor) -> bool:
        return resistance == resistor.r

    resistor = Implementation(Resistor, [ohms_law])
    with pytest.raises(BiltImplementationDimensionalityError) as raised:
        Implementation(Component, [resistance_eq], internal_implementations={"resistor": resistor})

    assert "makes 'resistor' dimensionally inconsistent" in str(raised.value)
    assert f"'resistor.r': {LENGTH}: inferred by resistance_eq from 'resistance'" in str(raised.value)


def test_the_parent_resolves_an_unconstrained_child_port() -> None:
    @system
    class Component(metaclass=Interface):
        voltage: Voltage = field()
        current: Current = field()
        resistance: Quantity = field()

    @equation
    def voltage_eq(voltage: Quantity, resistor: UnconstrainedResistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: Quantity, resistor: UnconstrainedResistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: Quantity, resistor: UnconstrainedResistor) -> bool:
        return resistance == resistor.r

    resistor = Implementation(UnconstrainedResistor, [ohms_law])
    with Implementation(
        Component, [voltage_eq, current_eq, resistance_eq], internal_implementations={"resistor": resistor}
    ):
        result = Component(voltage=6.0 * volt, current=2.0 * ampere).resistance

    assert result == 3.0 * ohm
    assert result.unit.dimension == RESISTANCE


def test_a_parent_making_the_child_inconsistent_raises() -> None:
    @system
    class Component(metaclass=Interface):
        voltage: Voltage = field()
        current: Current = field()
        resistance: Length = field()

    @equation
    def voltage_eq(voltage: Quantity, resistor: UnconstrainedResistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: Quantity, resistor: UnconstrainedResistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: Quantity, resistor: UnconstrainedResistor) -> bool:
        return resistance == resistor.r

    resistor = Implementation(UnconstrainedResistor, [ohms_law])
    with pytest.raises(BiltImplementationDimensionalityError) as raised:
        Implementation(
            Component, [voltage_eq, current_eq, resistance_eq], internal_implementations={"resistor": resistor}
        )

    assert "inferred by resistor from 'resistor.i', 'resistor.r'" in str(raised.value)


def test_a_dimensionality_a_grandchild_infers_reaches_the_grandparent() -> None:
    @system
    class Component(metaclass=Interface):
        resistance: Quantity = field()

    @equation
    def resistance_eq(resistance: Quantity, resistor: Resistor) -> bool:
        return resistance == resistor.r

    @system
    class Circuit(metaclass=Interface):
        total: Length = field()

    @equation
    def total_eq(total: Quantity, component: Component) -> bool:
        return total == component.resistance

    resistor = Implementation(Resistor, [ohms_law])
    component = Implementation(Component, [resistance_eq], internal_implementations={"resistor": resistor})
    with pytest.raises(BiltImplementationDimensionalityError) as raised:
        Implementation(Circuit, [total_eq], internal_implementations={"component": component})

    assert f"'component.resistance': {RESISTANCE}: inferred by component" in str(raised.value)


def test_a_dotted_input_no_parent_equation_reads_is_checked_against_the_child() -> None:
    @system
    class Component(metaclass=Interface):
        voltage: Quantity = field()

    @equation
    def voltage_eq(voltage: Quantity, resistor: Resistor) -> bool:
        return voltage == resistor.v

    resistor = Implementation(Resistor, [ohms_law])
    implementation = Implementation(Component, [voltage_eq], internal_implementations={"resistor": resistor})

    with pytest.raises(BiltInputDimensionalityError, match=r"'resistor\.i'"):
        implementation.evaluate({"resistor.i": 2.0 * meter, "resistor.r": 3.0 * ohm}, "voltage", PYTHON_BACKEND)


def test_a_dotted_quantity_output_given_only_floats_answers_with_a_quantity() -> None:
    @system
    class Component(metaclass=Interface):
        voltage: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    resistor = Implementation(Resistor, [ohms_law])
    implementation = Implementation(Component, [voltage_eq], internal_implementations={"resistor": resistor})

    assert implementation.evaluate({"voltage": 6.0}, "resistor.v", PYTHON_BACKEND) == 6.0 * volt
