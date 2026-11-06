"""Execution, compilation, and caching through nested subsystems."""

from typing import TYPE_CHECKING

import pytest

from bilt import Implementation, Interface, field, system, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._sub_system import SubSystemConstituent

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend

CURRENT = 2.0
RESISTANCE = 3.0
VOLTAGE = 6.0


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (
        ({"current": CURRENT, "resistance": RESISTANCE}, "voltage", VOLTAGE),
        ({"voltage": VOLTAGE, "resistance": RESISTANCE}, "current", CURRENT),
        ({"voltage": VOLTAGE, "current": CURRENT}, "resistance", RESISTANCE),
    ),
)
@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_parent_port_is_solved_in_any_direction_through_its_sub_system(
    backend: Backend,
    inputs: dict[str, float],
    output: str,
    expected: float,
    component_over_resistor: Implementation,
) -> None:
    instance = component_over_resistor.system(**inputs)
    with backend, component_over_resistor:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_an_outer_transform_consumes_a_port_the_sub_system_produced(backend: Backend) -> None:
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
    class Circuit(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()
        power: float = field()

    @transform
    def power(voltage: float, current: float) -> float:
        return voltage * current

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: float, resistor: Resistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: float, resistor: Resistor) -> bool:
        return resistance == resistor.r

    circuit = Implementation(
        Circuit,
        [voltage_eq, current_eq, resistance_eq, power],
        internal_implementations={"resistor": resistor},
    )

    instance = Circuit(current=CURRENT, resistance=RESISTANCE)
    with backend, circuit:
        assert instance.power == VOLTAGE * CURRENT


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_two_sub_systems_of_one_interface_are_solved_independently(
    backend: Backend, series_circuit_over_resistors: Implementation
) -> None:
    instance = series_circuit_over_resistors.system(i=CURRENT, r1=RESISTANCE, r2=RESISTANCE + 1)
    with backend, series_circuit_over_resistors:
        assert instance.v1 == CURRENT * RESISTANCE
        assert instance.v2 == CURRENT * (RESISTANCE + 1)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_sub_system_two_levels_down_is_solved(backend: Backend) -> None:
    """A ``Resistor`` inside a ``Component`` inside a ``Circuit``."""

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

    component = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor},
    )

    @system
    class Circuit(metaclass=Interface):
        total_voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def total_voltage_eq(total_voltage: float, component: Component) -> bool:
        return total_voltage == component.voltage

    @equation
    def circuit_current_eq(current: float, component: Component) -> bool:
        return current == component.current

    @equation
    def circuit_resistance_eq(resistance: float, component: Component) -> bool:
        return resistance == component.resistance

    circuit = Implementation(
        Circuit,
        [total_voltage_eq, circuit_current_eq, circuit_resistance_eq],
        internal_implementations={"component": component},
    )

    instance = Circuit(current=CURRENT, resistance=RESISTANCE)
    with backend, circuit:
        assert instance.total_voltage == VOLTAGE


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_child_port_is_solved_from_the_parents_own_inputs(
    backend: Backend, component_over_resistor: Implementation
) -> None:
    """The target is a port of the child, named as the parent sees it."""
    result = component_over_resistor.evaluate({"voltage": VOLTAGE, "resistance": RESISTANCE}, "resistor.i", backend)

    assert result == CURRENT


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_parent_port_is_solved_from_dotted_child_inputs(
    backend: Backend, component_over_resistor: Implementation
) -> None:
    result = component_over_resistor.evaluate({"resistor.i": CURRENT, "resistor.r": RESISTANCE}, "voltage", backend)

    assert result == VOLTAGE


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_child_port_no_equation_mentions_does_not_block_analysis(backend: Backend) -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()
        p: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @equation
    def power_eq(p: float, v: float, i: float) -> bool:
        return p == v * i

    resistor = Implementation(Resistor, [ohms_law, power_eq])

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

    component = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor},
    )

    instance = Component(current=CURRENT, resistance=RESISTANCE)
    with backend, component:
        assert instance.voltage == VOLTAGE


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_child_implementation_shared_by_two_references_is_compiled_once(
    backend: Backend, series_circuit_over_resistors: Implementation
) -> None:
    """Both references ask the child for the same route, so the child compiles it once."""
    constituent = series_circuit_over_resistors._components["resistor1"]  # noqa: SLF001
    assert isinstance(constituent, SubSystemConstituent)
    resistor = constituent.implementation

    instance = series_circuit_over_resistors.system(i=CURRENT, r1=RESISTANCE, r2=RESISTANCE + 1)
    with backend, series_circuit_over_resistors:
        _ = instance.v1
        _ = instance.v2

    assert len(resistor._evaluation_cache) == 1  # noqa: SLF001


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_repeated_evaluation_reuses_the_parents_one_cache_entry(
    backend: Backend, component_over_resistor: Implementation
) -> None:
    instance = component_over_resistor.system(current=CURRENT, resistance=RESISTANCE)
    with backend, component_over_resistor:
        first = instance.voltage
        second = instance.voltage

    assert first == second == VOLTAGE
    assert len(component_over_resistor._evaluation_cache) == 1  # noqa: SLF001


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_the_named_implementation_wins_over_the_active_default(backend: Backend) -> None:
    """A sub-system is bound to the implementation it was declared with, not to whatever is in scope."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @equation
    def doubled_ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r * 2.0

    declared = Implementation(Resistor, [ohms_law])
    in_scope = Implementation(Resistor, [doubled_ohms_law])

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

    component = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": declared},
    )

    instance = Component(current=CURRENT, resistance=RESISTANCE)
    with backend, in_scope, component:
        assert instance.voltage == VOLTAGE
