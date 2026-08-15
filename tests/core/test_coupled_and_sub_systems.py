from typing import TYPE_CHECKING

import pytest

from bilt._backend._jax._backend import JAXBackend
from bilt._backend._python._backend import PythonBackend
from bilt.core import Implementation, Interface, field, system, transform
from bilt.core._equation import equation

if TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_coupled_equations(
    backend: Backend,
) -> None:
    @system
    class Coupled(metaclass=Interface):
        a: float = field()
        b: float = field()
        x: float = field()
        y: float = field()

    @equation
    def first(x: float, y: float, a: float) -> bool:
        return x == a + y

    @equation
    def second(y: float, x: float, b: float) -> bool:
        return y == b - 0.5 * x

    implementation = Implementation(
        Coupled,
        [first, second],
    )

    instance = Coupled(
        a=1.0,
        b=4.0,
    )

    with backend, implementation:
        assert instance.x == pytest.approx(10.0 / 3.0)
        assert instance.y == pytest.approx(7.0 / 3.0)


@pytest.mark.parametrize(
    ("inputs", "output", "expected"),
    (({"current": 2.0, "resistance": 3.0}, "power", 12.0),),
)
@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_sub_system_output_consumed_by_outer_transform(
    backend: Backend,
    inputs: dict[str, float],
    output: str,
    expected: float,
) -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(
        Resistor,
        [ohms_law],
    )

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
    def resistance_eq(
        resistance: float,
        resistor: Resistor,
    ) -> bool:
        return resistance == resistor.r

    circuit_impl = Implementation(
        Circuit,
        [
            voltage_eq,
            current_eq,
            resistance_eq,
            power,
        ],
        internal_implementations={
            "resistor": resistor_impl,
        },
    )

    instance = Circuit(**inputs)

    with backend, circuit_impl:
        assert getattr(instance, output) == expected


@pytest.mark.parametrize(
    "backend",
    (PythonBackend(), JAXBackend()),
)
def test_evaluation_with_coupled_sub_system(
    backend: Backend,
) -> None:
    @system
    class CoupledChild(metaclass=Interface):
        a: float = field()
        b: float = field()
        x: float = field()
        y: float = field()

    @equation
    def first(x: float, y: float, a: float) -> bool:
        return x == a + y

    @equation
    def second(y: float, x: float, b: float) -> bool:
        return y == b - 0.5 * x

    child_impl = Implementation(
        CoupledChild,
        [first, second],
    )

    @system
    class Parent(metaclass=Interface):
        a: float = field()
        b: float = field()
        result: float = field()

    @equation
    def child_a(a: float, child: CoupledChild) -> bool:
        return a == child.a

    @equation
    def child_b(b: float, child: CoupledChild) -> bool:
        return b == child.b

    @equation
    def result(result: float, child: CoupledChild) -> bool:
        return result == child.x

    parent_impl = Implementation(
        Parent,
        [
            child_a,
            child_b,
            result,
        ],
        internal_implementations={
            "child": child_impl,
        },
    )

    instance = Parent(
        a=1.0,
        b=4.0,
    )

    with backend, parent_impl:
        assert instance.result == pytest.approx(10.0 / 3.0)
