from typing import TYPE_CHECKING

import pytest

from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core import (
    BiltSubSystemConflictError,
    BiltSubSystemError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
    BiltSystemAnalysisError,
    Implementation,
    Interface,
    field,
    system,
    transform,
)
from bilt.core._equation import equation
from bilt.core._sub_system import SubSystemConstituent

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent

CURRENT = 2.0
RESISTANCE = 3.0
VOLTAGE = 6.0


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


# --- wiring a reference up to an implementation ------------------------------------------------


def test_a_named_implementation_becomes_a_sub_system_constituent(
    component_over_resistor_voltage_only: Implementation,
) -> None:
    constituent = component_over_resistor_voltage_only._components["resistor"]  # noqa: SLF001

    assert isinstance(constituent, SubSystemConstituent)
    assert constituent.__bilt_name__ == "resistor"
    assert constituent.__bilt_parameters__ == ("resistor.v", "resistor.i", "resistor.r")


def test_two_references_to_one_implementation_become_separate_constituents(
    series_circuit_over_resistors: Implementation,
) -> None:
    first = series_circuit_over_resistors._components["resistor1"]  # noqa: SLF001
    second = series_circuit_over_resistors._components["resistor2"]  # noqa: SLF001

    assert isinstance(first, SubSystemConstituent)
    assert isinstance(second, SubSystemConstituent)
    assert first is not second
    assert first.implementation is second.implementation


def test_a_constituent_handed_in_whole_satisfies_the_reference() -> None:
    """``internal_implementations`` is the shorthand; a constituent built by hand is still a constituent."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @system
    class Board(metaclass=Interface):
        supply_voltage: float = field()

    @equation
    def supply_is_across_the_resistor(supply_voltage: float, resistor: Resistor) -> bool:
        return supply_voltage == resistor.v

    resistor = SubSystemConstituent(name="resistor", implementation=Implementation(Resistor, [ohms_law]))
    board = Implementation(Board, [supply_is_across_the_resistor, resistor])

    assert board._components["resistor"] is resistor  # noqa: SLF001


def test_an_unreferenced_constituent_handed_in_whole_is_kept() -> None:
    """A constituent named as a component is a component, whether or not an equation mentions it."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    @system
    class Board(metaclass=Interface):
        supply_voltage: float = field()

    resistor = SubSystemConstituent(name="resistor", implementation=Implementation(Resistor, [ohms_law]))

    assert Implementation(Board, [resistor])._components["resistor"] is resistor  # noqa: SLF001


# --- rejecting a mis-declared sub-system --------------------------------------------------------


def test_a_reference_with_no_implementation_is_rejected() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    with pytest.raises(
        BiltSubSystemMissingImplementationError,
        match=r"Sub-system reference 'resistor'.*has no entry in internal_implementations",
    ):
        Implementation(Component, [voltage_eq], internal_implementations={})


def test_a_reference_annotated_but_never_read_still_needs_an_implementation() -> None:
    """The annotation is the reference; an equation need not dereference a port to depend on the child."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Board(metaclass=Interface):
        voltage: float = field()
        supply: float = field()

    @equation
    def voltage_eq(voltage: float, supply: float, resistor: Resistor) -> bool:  # noqa: ARG001
        return voltage == supply

    assert voltage_eq.__bilt_parameters__ == ("voltage", "supply")

    with pytest.raises(BiltSubSystemMissingImplementationError, match="Sub-system reference 'resistor'"):
        Implementation(Board, [voltage_eq])


def test_an_implementation_of_the_wrong_interface_is_rejected() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Capacitor(metaclass=Interface):
        q: float = field()
        c: float = field()

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def charge(q: float, c: float) -> bool:
        return q == c

    capacitor = Implementation(Capacitor, [charge])

    with pytest.raises(
        BiltSubSystemMismatchedImplementationError,
        match=r"Sub-system reference 'resistor' expects interface 'Resistor' but.*has interface 'Capacitor'",
    ) as raised:
        Implementation(Component, [voltage_eq], internal_implementations={"resistor": capacitor})

    assert "share a name" not in str(raised.value)


def test_an_implementation_of_the_wrong_interface_annotation() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Thermistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    assert Thermistor.__bilt_parameters__ == Resistor.__bilt_parameters__

    @system
    class Component(metaclass=Interface):
        voltage: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @system
    class Circuit(metaclass=Interface):
        total_voltage: float = field()

    @equation
    def total_voltage_eq(total_voltage: float, component: Component) -> bool:
        return total_voltage == component.voltage

    with pytest.raises(
        BiltSubSystemMismatchedImplementationError,
        match=r"Sub-system reference 'resistor' expects interface 'Resistor' but.*has interface 'Thermistor'",
    ) as raised:
        Implementation(
            Circuit,
            [total_voltage_eq],
            internal_implementations={
                "component": Implementation(
                    Component, [voltage_eq], internal_implementations={"resistor": Implementation(Thermistor, [])}
                )
            },
        )

    assert raised.value.reference_name == "resistor"


def test_an_implementation_of_a_look_alike_interface_says_why_the_names_match() -> None:
    """Interfaces are matched by identity, so a bare name-vs-name message would read as a bug."""

    def board_annotated_with_its_own_resistor() -> tuple[Interface, BiltModelConstituent]:
        @system
        class Resistor(metaclass=Interface):
            v: float = field()
            i: float = field()
            r: float = field()

        @system
        class Board(metaclass=Interface):
            supply: float = field()

        @equation
        def wiring(supply: float, resistor: Resistor) -> bool:
            return supply == resistor.v

        return Board, wiring

    def a_separately_declared_resistor() -> Implementation:
        @system
        class Resistor(metaclass=Interface):
            v: float = field()
            i: float = field()
            r: float = field()

        return Implementation(Resistor, [])

    board, wiring = board_annotated_with_its_own_resistor()

    with pytest.raises(BiltSubSystemMismatchedImplementationError, match="separate classes that happen to share"):
        Implementation(board, [wiring], internal_implementations={"resistor": a_separately_declared_resistor()})


def test_one_reference_used_with_two_look_alike_interfaces_says_why_the_names_match() -> None:
    def x_equation_over_its_own_resistor() -> BiltModelConstituent:
        @system
        class Resistor(metaclass=Interface):
            v: float = field()
            i: float = field()
            r: float = field()

        @equation
        def x_eq(x: float, sub: Resistor) -> bool:
            return x == sub.v

        return x_eq

    def y_equation_over_its_own_resistor() -> BiltModelConstituent:
        @system
        class Resistor(metaclass=Interface):
            v: float = field()
            i: float = field()
            r: float = field()

        @equation
        def y_eq(y: float, sub: Resistor) -> bool:
            return y == sub.v

        return y_eq

    @system
    class Parent(metaclass=Interface):
        x: float = field()
        y: float = field()

    with pytest.raises(BiltSubSystemConflictError, match="separate classes that happen to share"):
        Implementation(Parent, [x_equation_over_its_own_resistor(), y_equation_over_its_own_resistor()])


def test_an_implementation_no_equation_references_is_rejected() -> None:
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

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    with pytest.raises(
        BiltSubSystemUnreferencedEntryError,
        match=r"internal_implementations has entry 'unused'.*no equation references it",
    ):
        Implementation(
            Component,
            [voltage_eq],
            internal_implementations={"resistor": resistor, "unused": resistor},
        )


def test_a_reference_named_after_a_component_is_rejected() -> None:
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

    @transform
    def voltage(current: float, resistance: float) -> float:
        return current * resistance

    @equation
    def voltage_eq(supply: float, voltage: Resistor) -> bool:
        return supply == voltage.v

    with pytest.raises(
        BiltSubSystemNameCollisionError,
        match="Sub-system reference 'voltage' collides with a component name",
    ):
        Implementation(Component, [voltage, voltage_eq], internal_implementations={"voltage": resistor})


def test_every_sub_system_declaration_error_is_catchable_as_one_category() -> None:
    """A caller who only wants "the nesting does not add up" should not have to name all five."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Board(metaclass=Interface):
        supply: float = field()

    @equation
    def wiring(supply: float, resistor: Resistor) -> bool:
        return supply == resistor.v

    with pytest.raises(BiltSubSystemError) as raised:
        Implementation(Board, [wiring])

    assert isinstance(raised.value, BiltSubSystemMissingImplementationError)
    assert raised.value.reference_name == "resistor"


def test_one_reference_used_with_two_interfaces_is_rejected() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Capacitor(metaclass=Interface):
        q: float = field()
        c: float = field()

    @system
    class Parent(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def x_eq(x: float, sub: Resistor) -> bool:
        return x == sub.v

    @equation
    def y_eq(y: float, sub: Capacitor) -> bool:
        return y == sub.q

    with pytest.raises(
        BiltSubSystemConflictError,
        match="Sub-system reference 'sub' is used with different interfaces",
    ):
        Implementation(Parent, [x_eq, y_eq])


# --- evaluating through a sub-system ------------------------------------------------------------


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


# --- refusing to guess ---------------------------------------------------------------------------


def test_an_under_specified_sub_system_is_not_guessed_at(
    component_over_resistor_voltage_only: Implementation,
) -> None:
    """Only the voltage port is bridged, so current and resistance reach nothing."""
    instance = component_over_resistor_voltage_only.system(current=CURRENT, resistance=RESISTANCE)
    with component_over_resistor_voltage_only, pytest.raises(BiltSystemAnalysisError):
        _ = instance.voltage


def test_a_direction_the_child_cannot_solve_is_rejected() -> None:
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

    component = Implementation(
        Component,
        [voltage_eq, current_eq],
        internal_implementations={"resistor": resistor},
    )

    instance = Component(voltage=VOLTAGE, resistance=RESISTANCE)
    with component, pytest.raises(BiltSystemAnalysisError):
        _ = instance.current
