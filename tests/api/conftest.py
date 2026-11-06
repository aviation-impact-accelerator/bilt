"""Shared nested-subsystem model fixtures for API tests."""

import pytest

from bilt import Implementation, Interface, field, system
from bilt.core._equation import equation


@pytest.fixture
def component_over_resistor() -> Implementation:
    """A ``Component`` whose three ports are each bridged to a ``Resistor`` sub-system."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: float, resistor: Resistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: float, resistor: Resistor) -> bool:
        return resistance == resistor.r

    return Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor},
    )


@pytest.fixture
def component_over_resistor_voltage_only() -> Implementation:
    """A ``Component`` bridged to its ``Resistor`` sub-system through the voltage port alone."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    return Implementation(Component, [voltage_eq], internal_implementations={"resistor": resistor})


@pytest.fixture
def series_circuit_over_resistors() -> Implementation:
    """A ``SeriesCircuit`` bridged to two references sharing one ``Resistor`` implementation."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor = Implementation(Resistor, [ohms_law])

    @system
    class SeriesCircuit(metaclass=Interface):
        v1: float = field()
        v2: float = field()
        i: float = field()
        r1: float = field()
        r2: float = field()

    @equation
    def v1_eq(v1: float, resistor1: Resistor) -> bool:
        return v1 == resistor1.v

    @equation
    def v2_eq(v2: float, resistor2: Resistor) -> bool:
        return v2 == resistor2.v

    @equation
    def i1_eq(i: float, resistor1: Resistor) -> bool:
        return i == resistor1.i

    @equation
    def r1_eq(r1: float, resistor1: Resistor) -> bool:
        return r1 == resistor1.r

    @equation
    def i2_eq(i: float, resistor2: Resistor) -> bool:
        return i == resistor2.i

    @equation
    def r2_eq(r2: float, resistor2: Resistor) -> bool:
        return r2 == resistor2.r

    return Implementation(
        SeriesCircuit,
        [v1_eq, v2_eq, i1_eq, r1_eq, i2_eq, r2_eq],
        internal_implementations={"resistor1": resistor, "resistor2": resistor},
    )


@pytest.fixture
def component_over_sensed_resistor() -> Implementation:
    """A ``Component`` bridged to a ``Resistor`` whose current is reachable only through its sense port."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()
        i_sense: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @equation
    def sense_eq(i_sense: float, i: float) -> bool:
        return i_sense == i

    resistor = Implementation(Resistor, [ohms_law, sense_eq])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        sense_current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def sense_current_eq(sense_current: float, resistor: Resistor) -> bool:
        return sense_current == resistor.i_sense

    @equation
    def resistance_eq(resistance: float, resistor: Resistor) -> bool:
        return resistance == resistor.r

    return Implementation(
        Component,
        [voltage_eq, sense_current_eq, resistance_eq],
        internal_implementations={"resistor": resistor},
    )
