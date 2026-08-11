from typing import TYPE_CHECKING, NamedTuple

import pytest

from bilt._backend._jax._backend import JAXBackend
from bilt._backend._python._backend import PythonBackend
from bilt.core import (
    BiltSubSystemConflictError,
    BiltSubSystemMismatchedImplementationError,
    BiltSubSystemMissingImplementationError,
    BiltSubSystemNameCollisionError,
    BiltSubSystemUnreferencedEntryError,
    BiltSystemAnalysisError,
    Implementation,
    Interface,
    SubSystemConstituent,
    field,
    system,
    transform,
)
from bilt.core._equation import equation
from bilt.core._structural_analysis._exhaustive._bipartite import required_inputs_for_target
from bilt.core._structural_analysis._plan import PlanStep

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class NestedImplementation(NamedTuple):
    """An outer implementation together with the inner implementation it embeds."""

    outer: Implementation
    inner: Implementation


@pytest.fixture
def component_over_resistor() -> NestedImplementation:
    """A `Component` whose three ports are each bridged to a `Resistor` sub-system."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

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

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor_impl},
    )
    return NestedImplementation(outer=component_impl, inner=resistor_impl)


@pytest.fixture
def component_over_resistor_voltage_only() -> NestedImplementation:
    """A `Component` bridged to its `Resistor` sub-system through the voltage port alone."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    component_impl = Implementation(
        Component,
        [voltage_eq],
        internal_implementations={"resistor": resistor_impl},
    )
    return NestedImplementation(outer=component_impl, inner=resistor_impl)


@pytest.fixture
def series_circuit_over_resistors() -> NestedImplementation:
    """A `SeriesCircuit` bridged to two references sharing one `Resistor` implementation."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

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

    series_impl = Implementation(
        SeriesCircuit,
        [v1_eq, v2_eq, i1_eq, r1_eq, i2_eq, r2_eq],
        internal_implementations={"resistor1": resistor_impl, "resistor2": resistor_impl},
    )
    return NestedImplementation(outer=series_impl, inner=resistor_impl)


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (({"current": 2.0, "resistance": 3.0}, "voltage", 6.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_basic(
    backend: Backend,
    inputs: dict[str, float],
    output: str,
    expected: float,
    component_over_resistor: NestedImplementation,
) -> None:
    component_impl = component_over_resistor.outer

    instance = component_impl.system(**inputs)
    with backend, component_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (
        ({"current": 2.0, "resistance": 3.0}, "voltage", 6.0),
        ({"voltage": 6.0, "resistance": 3.0}, "current", 2.0),
        ({"voltage": 6.0, "current": 2.0}, "resistance", 3.0),
    ),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_bidirectional(
    backend: Backend,
    inputs: dict[str, float],
    output: str,
    expected: float,
    component_over_resistor: NestedImplementation,
) -> None:
    component_impl = component_over_resistor.outer

    instance = component_impl.system(**inputs)
    with backend, component_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (({"current": 2.0, "resistance": 3.0}, "power", 12.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_output_consumed_by_outer_transform(
    backend: Backend, inputs: dict[str, float], output: str, expected: float
) -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

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

    circuit_impl = Implementation(
        Circuit,
        [voltage_eq, current_eq, resistance_eq, power],
        internal_implementations={"resistor": resistor_impl},
    )

    instance = Circuit(**inputs)
    with backend, circuit_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    ("inputs", "expected_v1", "expected_v2"),
    (({"i": 2.0, "r1": 3.0, "r2": 4.0}, 6.0, 8.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_two_sub_systems_of_same_type(
    backend: Backend,
    inputs: dict[str, float],
    expected_v1: float,
    expected_v2: float,
    series_circuit_over_resistors: NestedImplementation,
) -> None:
    series_impl = series_circuit_over_resistors.outer

    instance = series_impl.system(**inputs)
    with backend, series_impl:
        assert instance.v1 == expected_v1
        assert instance.v2 == expected_v2


def test_sub_system_missing_implementation() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    with pytest.raises(
        BiltSubSystemMissingImplementationError,
        match=r"Sub-system reference 'resistor'.*has no entry in internal_implementations",
    ):
        Implementation(Component, [voltage_eq], internal_implementations={})


def test_sub_system_mismatched_implementation() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class OtherSystem(metaclass=Interface):
        x: float = field()
        y: float = field()

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def dummy(x: float, y: float) -> bool:
        return x == y

    other_impl = Implementation(OtherSystem, [dummy])

    with pytest.raises(
        BiltSubSystemMismatchedImplementationError,
        match=r"Sub-system reference 'resistor' expects interface 'Resistor' but.*has interface 'OtherSystem'",
    ):
        Implementation(Component, [voltage_eq], internal_implementations={"resistor": other_impl})


def test_sub_system_unreferenced_entry() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

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
            internal_implementations={"resistor": resistor_impl, "unused": resistor_impl},
        )


def test_sub_system_name_collision() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @transform
    def voltage(current: float, resistance: float) -> float:
        return current * resistance

    @equation
    def voltage_eq(voltage_out: float, voltage: Resistor) -> bool:
        return voltage_out == voltage.v

    with pytest.raises(
        BiltSubSystemNameCollisionError,
        match="Sub-system reference 'voltage' collides with a component name",
    ):
        Implementation(
            Component,
            [voltage, voltage_eq],
            internal_implementations={"voltage": resistor_impl},
        )


def test_sub_system_conflict_different_interfaces() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @system
    class OtherComponent(metaclass=Interface):
        a: float = field()
        b: float = field()

    @system
    class Parent(metaclass=Interface):
        x: float = field()
        y: float = field()
        z: float = field()

    @equation
    def eq1(x: float, sub: Resistor) -> bool:
        return x == sub.v

    @equation
    def eq2(y: float, sub: OtherComponent) -> bool:
        return y == sub.a

    with pytest.raises(
        BiltSubSystemConflictError,
        match="Sub-system reference 'sub' is used with different interfaces",
    ):
        Implementation(Parent, [eq1, eq2])


def test_sub_system_correct_specification(component_over_resistor_voltage_only: NestedImplementation) -> None:
    component_impl = component_over_resistor_voltage_only.outer

    assert "resistor" in component_impl._components  # noqa: SLF001
    constituent = component_impl._components["resistor"]  # noqa: SLF001
    assert isinstance(constituent, SubSystemConstituent)
    assert constituent.__bilt_name__ == "resistor"
    assert constituent.__bilt_parameters__ == ("resistor.v", "resistor.i", "resistor.r")


def test_sub_system_two_references_same_type_different_constituents(
    series_circuit_over_resistors: NestedImplementation,
) -> None:
    series_impl = series_circuit_over_resistors.outer

    constituent1 = series_impl._components["resistor1"]  # noqa: SLF001
    constituent2 = series_impl._components["resistor2"]  # noqa: SLF001

    assert isinstance(constituent1, SubSystemConstituent)
    assert isinstance(constituent2, SubSystemConstituent)
    assert constituent1 != constituent2


def test_sub_system_constituent_required_inputs_basic(
    component_over_resistor_voltage_only: NestedImplementation,
) -> None:
    component_impl = component_over_resistor_voltage_only.outer

    resistor_constituent = component_impl._components["resistor"]  # noqa: SLF001
    assert isinstance(resistor_constituent, SubSystemConstituent)

    required = required_inputs_for_target(PlanStep(operation=resistor_constituent, output="resistor.v"))
    assert required == frozenset({"resistor.i", "resistor.r"})


def test_sub_system_constituent_required_inputs_no_path() -> None:
    @system
    class EmptySystem(metaclass=Interface):
        a: float = field()
        b: float = field()

    empty_impl = Implementation(EmptySystem, [])

    @system
    class Component(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def dummy_eq(x: float, subsys: EmptySystem) -> bool:
        return x == subsys.a

    component_impl = Implementation(
        Component,
        [dummy_eq],
        internal_implementations={"subsys": empty_impl},
    )

    subsys_constituent = component_impl._components["subsys"]  # noqa: SLF001
    assert isinstance(subsys_constituent, SubSystemConstituent)

    required = required_inputs_for_target(PlanStep(operation=subsys_constituent, output="subsys.a"))
    assert required is None


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (({"current": 2.0, "resistance": 3.0}, "total_voltage", 6.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_two_levels_deep(
    backend: Backend, inputs: dict[str, float], output: str, expected: float
) -> None:
    """Resistor inside Component inside Circuit."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

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

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor_impl},
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
    def current_eq_circuit(current: float, component: Component) -> bool:
        return current == component.current

    @equation
    def resistance_eq_circuit(resistance: float, component: Component) -> bool:
        return resistance == component.resistance

    circuit_impl = Implementation(
        Circuit,
        [total_voltage_eq, current_eq_circuit, resistance_eq_circuit],
        internal_implementations={"component": component_impl},
    )

    instance = Circuit(**inputs)
    with backend, circuit_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_solved_inward(
    backend: Backend, component_over_resistor: NestedImplementation
) -> None:
    """CHILD port is the target."""
    component_impl = component_over_resistor.outer

    inputs: dict[str, object] = {"voltage": 6.0, "resistance": 3.0}
    expected_child_current = 2.0
    with backend, component_impl:
        result = component_impl.evaluate(inputs, "resistor.i", backend)
        assert result == expected_child_current

    dotted_inputs: dict[str, object] = {"resistor.i": 2.0, "resistor.r": 3.0}
    expected_voltage = 6.0
    with backend, component_impl:
        result = component_impl.evaluate(dotted_inputs, "voltage", backend)
        assert result == expected_voltage


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (({"current": 2.0, "resistance": 3.0}, "voltage", 6.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_unreferenced_port(
    backend: Backend, inputs: dict[str, float], output: str, expected: float
) -> None:
    """Unreferenced ports don't break analysis."""

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

    resistor_impl = Implementation(Resistor, [ohms_law, power_eq])

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

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor_impl},
    )

    instance = Component(**inputs)
    with backend, component_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_sub_system_shared_implementation_uses_one_cache_entry(
    backend: Backend, series_circuit_over_resistors: NestedImplementation
) -> None:
    """Shared implementation uses one cache entry."""
    series_impl = series_circuit_over_resistors.outer
    resistor_impl = series_circuit_over_resistors.inner

    instance = series_impl.system(i=2.0, r1=3.0, r2=4.0)
    with backend, series_impl:
        _ = instance.v1
        _ = instance.v2

    assert len(resistor_impl._callable_cache) == 1  # noqa: SLF001


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (({"current": 2.0, "resistance": 3.0}, "voltage", 6.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_sub_system_implementation_selected_explicitly(
    backend: Backend, inputs: dict[str, float], output: str, expected: float
) -> None:
    """Implementation is selected explicitly from internal_implementations."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law_normal(v: float, i: float, r: float) -> bool:
        return v == i * r

    @equation
    def ohms_law_doubled(v: float, i: float, r: float) -> bool:
        return v == i * r * 2.0

    resistor_impl_normal = Implementation(Resistor, [ohms_law_normal])
    resistor_impl_doubled = Implementation(Resistor, [ohms_law_doubled])

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

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor_impl_normal},
    )

    instance = Component(**inputs)
    with backend, resistor_impl_doubled, component_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    ("inputs", "expected_voltage"),
    (({"current": 2.0, "resistance": 3.0}, 6.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_sub_system_reuses_realized_plan(
    backend: Backend,
    inputs: dict[str, float],
    expected_voltage: float,
    component_over_resistor: NestedImplementation,
) -> None:
    """Realized plan is reused on second evaluation."""
    component_impl = component_over_resistor.outer

    instance = component_impl.system(**inputs)
    with backend, component_impl:
        v1 = instance.voltage
        v2 = instance.voltage

    assert v1 == v2 == expected_voltage
    assert len(component_impl._callable_cache) == 1  # noqa: SLF001


def test_under_specified_sub_system_raises(component_over_resistor_voltage_only: NestedImplementation) -> None:
    """Heuristic port mapping no longer silently guesses connections."""
    component_impl = component_over_resistor_voltage_only.outer
    resistor_impl = component_over_resistor_voltage_only.inner

    instance = component_impl.system(current=2.0, resistance=3.0)
    with resistor_impl, component_impl, pytest.raises(BiltSystemAnalysisError):
        _ = instance.voltage


def test_unknown_sub_system_port_raises() -> None:
    """Unknown sub-system port raises AttributeError."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    with pytest.raises(AttributeError, match="has no port 'typo'"):

        @equation
        def voltage_eq(voltage: float, resistor: Resistor) -> bool:
            return voltage == resistor.typo  # pyrefly: ignore[missing-attribute]  # ty: ignore[unresolved-attribute]


def test_child_cannot_solve_direction_raises() -> None:
    """Child cannot solve unsolvable direction."""

    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

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

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq],
        internal_implementations={"resistor": resistor_impl},
    )

    instance = Component(voltage=10.0, resistance=3.0)
    with resistor_impl, component_impl, pytest.raises(BiltSystemAnalysisError):
        _ = instance.current
