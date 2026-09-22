from typing import TYPE_CHECKING, Any

import pytest

from bilt import (
    Implementation,
    Interface,
    field,
    system,
    transform,
)
from bilt.backend._jax._backend import JAX_BACKEND
from bilt.backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._sub_system import SubSystemConstituent
from bilt.errors import (
    BiltAnalysisError,
    BiltSubSystemConflictError,
    BiltSubSystemError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
)

if TYPE_CHECKING:
    from bilt.backend._base_backend import Backend
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


def _sensed_resistor() -> tuple[Any, Implementation]:
    """A ``Resistor`` whose current is reachable directly or through its sense port."""

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

    return Resistor, Implementation(Resistor, [ohms_law, sense_eq])


def _child_implementation(parent: Implementation, name: str) -> Implementation:
    """The implementation behind one of ``parent``'s sub-system references."""
    constituent = parent._components[name]  # noqa: SLF001
    assert isinstance(constituent, SubSystemConstituent)
    return constituent.implementation


@pytest.fixture
def component_over_sensed_resistor() -> Implementation:
    """A ``Component`` bridged to a ``Resistor`` whose current is reachable only through its sense port."""
    Resistor, resistor = _sensed_resistor()  # noqa: N806

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


def test_a_reference_annotated_but_never_read_does_not_need_an_implementation() -> None:
    """Usage is the reference; an annotation the equation never dereferences is not a dependency."""

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

    board = Implementation(Board, [voltage_eq])

    assert "resistor" not in board._components  # noqa: SLF001
    assert board.evaluate({"supply": VOLTAGE}, "voltage", PYTHON_BACKEND) == VOLTAGE


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


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_child_on_its_own_takes_the_route_its_given_inputs_reach(
    backend: Backend, component_over_sensed_resistor: Implementation
) -> None:
    """Handed the sense port instead of ``i``, the child solves ``r`` by the longer route."""
    resistor = _child_implementation(component_over_sensed_resistor, "resistor")

    instance = resistor.system(v=VOLTAGE, i_sense=CURRENT)
    with backend, resistor:
        assert instance.r == RESISTANCE


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_nested_child_takes_the_route_the_parents_inputs_reach(
    backend: Backend, component_over_sensed_resistor: Implementation
) -> None:
    """The route the child offers the parent asks for ``i``, which no parent port reaches."""
    instance = component_over_sensed_resistor.system(voltage=VOLTAGE, sense_current=CURRENT)
    with backend, component_over_sensed_resistor:
        assert instance.resistance == RESISTANCE


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_reachable_route_is_still_found_two_levels_down(
    backend: Backend, component_over_sensed_resistor: Implementation
) -> None:
    """The sensed resistor one level further in, so a three-level hierarchy behaves like a two."""
    component = component_over_sensed_resistor
    Component: Any = component.system  # noqa: N806

    @system
    class Rig(metaclass=Interface):
        supply: float = field()
        shunt: float = field()
        load: float = field()

    @equation
    def supply_eq(supply: float, component: Component) -> bool:
        return supply == component.voltage

    @equation
    def shunt_eq(shunt: float, component: Component) -> bool:
        return shunt == component.sense_current

    @equation
    def load_eq(load: float, component: Component) -> bool:
        return load == component.resistance

    rig = Implementation(
        Rig,
        [supply_eq, shunt_eq, load_eq],
        internal_implementations={"component": component},
    )

    instance = Rig(supply=VOLTAGE, shunt=CURRENT)
    with backend, rig:
        assert instance.load == RESISTANCE


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_two_parents_reach_one_shared_child_by_different_routes(backend: Backend) -> None:
    """One parent is wired to the resistor's current, the other only to its sense port."""
    Resistor, resistor = _sensed_resistor()  # noqa: N806

    @system
    class DirectRig(metaclass=Interface):
        across: float = field()
        through: float = field()
        ohms: float = field()

    @equation
    def direct_across_eq(across: float, resistor: Resistor) -> bool:
        return across == resistor.v

    @equation
    def direct_through_eq(through: float, resistor: Resistor) -> bool:
        return through == resistor.i

    @equation
    def direct_ohms_eq(ohms: float, resistor: Resistor) -> bool:
        return ohms == resistor.r

    direct = Implementation(
        DirectRig,
        [direct_across_eq, direct_through_eq, direct_ohms_eq],
        internal_implementations={"resistor": resistor},
    )

    @system
    class SensedRig(metaclass=Interface):
        across: float = field()
        sensed: float = field()
        ohms: float = field()

    @equation
    def sensed_across_eq(across: float, resistor: Resistor) -> bool:
        return across == resistor.v

    @equation
    def sensed_shunt_eq(sensed: float, resistor: Resistor) -> bool:
        return sensed == resistor.i_sense

    @equation
    def sensed_ohms_eq(ohms: float, resistor: Resistor) -> bool:
        return ohms == resistor.r

    sensed = Implementation(
        SensedRig,
        [sensed_across_eq, sensed_shunt_eq, sensed_ohms_eq],
        internal_implementations={"resistor": resistor},
    )

    direct_instance = DirectRig(across=VOLTAGE, through=CURRENT)
    sensed_instance = SensedRig(across=VOLTAGE, sensed=CURRENT)
    with backend, direct:
        assert direct_instance.ohms == RESISTANCE
    with backend, sensed:
        assert sensed_instance.ohms == RESISTANCE


def test_a_relation_one_backend_rejects_survives_for_a_backend_that_accepts_it() -> None:
    """A transpile failure belongs to the backend that raised it, so it must not narrow the model."""

    @system
    class Doubler(metaclass=Interface):
        raw: float = field()
        doubled: float = field()

    @transform
    def doubled(raw: float) -> float:
        total = raw
        total += raw  # an augmented assignment is fine in Python and refused by JAX
        return total

    doubler = Implementation(Doubler, [doubled])

    @system
    class Channel(metaclass=Interface):
        reading: float = field()
        scaled: float = field()

    @equation
    def reading_eq(reading: float, doubler: Doubler) -> bool:
        return reading == doubler.raw

    @equation
    def scaled_eq(scaled: float, doubler: Doubler) -> bool:
        return scaled == doubler.doubled

    channel = Implementation(Channel, [reading_eq, scaled_eq], internal_implementations={"doubler": doubler})

    rejected = Channel(reading=VOLTAGE)
    with channel, JAX_BACKEND, pytest.raises(BiltAnalysisError):
        _ = rejected.scaled

    accepted = Channel(reading=VOLTAGE)
    with channel, PYTHON_BACKEND:
        assert accepted.scaled == VOLTAGE + VOLTAGE


def _rig_over(component: Implementation) -> tuple[Interface, Implementation]:
    """A ``Rig`` wired to the three ports of ``component``'s system, returned outermost first."""
    Component: Any = component.system  # noqa: N806

    @system
    class Rig(metaclass=Interface):
        supply: float = field()
        draw: float = field()
        load: float = field()

    @equation
    def supply_eq(supply: float, component: Component) -> bool:
        return supply == component.voltage

    @equation
    def draw_eq(draw: float, component: Component) -> bool:
        return draw == component.current

    @equation
    def load_eq(load: float, component: Component) -> bool:
        return load == component.resistance

    return Rig, Implementation(
        Rig,
        [supply_eq, draw_eq, load_eq],
        internal_implementations={"component": component},
    )


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_building_a_parents_analyser_does_not_reach_down_the_hierarchy(
    backend: Backend, component_over_resistor: Implementation
) -> None:
    """A hierarchy costs what a query walks, not what it contains."""
    resistor = _child_implementation(component_over_resistor, "resistor")
    _, rig = _rig_over(component_over_resistor)

    with EvaluationContext(backend):
        rig.analyser()

    assert component_over_resistor._analysers == {}  # noqa: SLF001
    assert resistor._analysers == {}  # noqa: SLF001


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_query_that_walks_the_hierarchy_analyses_what_it_walks(
    backend: Backend, component_over_resistor: Implementation
) -> None:
    """The other half of laziness: what a query does reach is analysed, and cached once."""
    component = component_over_resistor
    resistor = _child_implementation(component, "resistor")
    rig_system, rig = _rig_over(component)

    instance = rig_system(supply=VOLTAGE, draw=CURRENT)
    with backend, rig:
        assert instance.load == RESISTANCE

    assert list(component._analysers) == [backend]  # noqa: SLF001
    assert list(resistor._analysers) == [backend]  # noqa: SLF001


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND))
def test_a_model_with_two_reachable_routes_reads_the_same_every_time(backend: Backend) -> None:
    """Which route wins is bilt's business; that identical code keeps picking it is not."""

    def build_panel() -> tuple[Interface, Implementation]:
        @system
        class Splitter(metaclass=Interface):
            out: float = field()
            a: float = field()
            b: float = field()

        @equation
        def from_a(out: float, a: float) -> bool:
            return out == a + 1.0

        @equation
        def from_b(out: float, b: float) -> bool:
            return out == b * 10.0

        splitter = Implementation(Splitter, [from_a, from_b])

        @system
        class Panel(metaclass=Interface):
            first: float = field()
            second: float = field()
            reading: float = field()

        @equation
        def first_eq(first: float, splitter: Splitter) -> bool:
            return first == splitter.a

        @equation
        def second_eq(second: float, splitter: Splitter) -> bool:
            return second == splitter.b

        @equation
        def reading_eq(reading: float, splitter: Splitter) -> bool:
            return reading == splitter.out

        return Panel, Implementation(
            Panel,
            [first_eq, second_eq, reading_eq],
            internal_implementations={"splitter": splitter},
        )

    readings: list[object] = []
    for _ in range(2):
        panel_system, panel = build_panel()
        instance = panel_system(first=CURRENT, second=RESISTANCE)
        with backend, panel:
            readings.append(instance.reading)

    assert readings[0] == readings[1]


# --- refusing to guess ---------------------------------------------------------------------------


def test_an_under_specified_sub_system_is_not_guessed_at(
    component_over_resistor_voltage_only: Implementation,
) -> None:
    """Only the voltage port is bridged, so current and resistance reach nothing."""
    instance = component_over_resistor_voltage_only.system(current=CURRENT, resistance=RESISTANCE)
    with component_over_resistor_voltage_only, pytest.raises(BiltAnalysisError):
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
    with component, pytest.raises(BiltAnalysisError):
        _ = instance.current


def test_a_python_transform_can_evaluate_another_implementation_under_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape hatch documented in `docs/guide/backends.md`.

    A transform running under one backend evaluates another implementation under another, so the
    nested result carries the nested backend's array type rather than the outer backend's float.
    """

    @system
    class JaxSubproblem(metaclass=Interface):
        value: float = field()
        result: float = field()

    @transform
    def result(value: float) -> float:
        return value**2

    subproblem = Implementation(JaxSubproblem, [result])

    @system
    class Outer(metaclass=Interface):
        input_value: float = field()
        output_value: float = field()

    @transform
    def output_value(input_value: float) -> float:
        with JAX_BACKEND, subproblem:
            return JaxSubproblem(value=input_value).result

    # The docs spell these as imports inside the body, because a transform is recompiled from
    # source into the backend's namespace. Locals of a test are not importable, so seed them.
    monkeypatch.setitem(PYTHON_BACKEND.compilation_namespace, "JAX_BACKEND", JAX_BACKEND)
    monkeypatch.setitem(PYTHON_BACKEND.compilation_namespace, "JaxSubproblem", JaxSubproblem)
    monkeypatch.setitem(PYTHON_BACKEND.compilation_namespace, "subproblem", subproblem)

    expected = 9.0
    with PYTHON_BACKEND, Implementation(Outer, [output_value]):
        nested = Outer(input_value=3.0).output_value

    assert float(nested) == expected
    assert type(nested).__module__.startswith("jaxlib")
