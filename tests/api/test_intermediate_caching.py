"""Instance evaluation retains reusable ports with their logical names, domains, and units."""

import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Quantity, Tensor, equation, field, system, tensor, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._system import _implementation as implementation_module
from bilt.core._system._sentinel import MISSING
from bilt.symbolic._dimensions import Dim, coordinate_axis
from bilt.symbolic._domains import domain
from bilt.units.dimensions import CURRENT, MASS, POWER, RESISTANCE, VOLTAGE
from bilt.units.units import ampere, gram, kilogram, ohm, volt, watt

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCallable, BackendCase
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._system._instance import SystemInstance
    from bilt.core._system._lowering_and_lifting import LiftingPlan

Mass = typing.Annotated[Quantity, MASS]
MassTensor = typing.Annotated[Tensor, MASS]
Current = typing.Annotated[Quantity, CURRENT]
Resistance = typing.Annotated[Quantity, RESISTANCE]
Voltage = typing.Annotated[Quantity, VOLTAGE]
Power = typing.Annotated[Quantity, POWER]


@pytest.fixture(autouse=True)
def _select_structural_analyser(analyser_factory: object) -> None:
    _ = analyser_factory


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return typing.cast("Backend", request.param)


def _variables(instance: object) -> dict[str, object]:
    return typing.cast("SystemInstance", instance).__bilt_system_instance_variables__


def test_later_requests_reuse_computed_ports(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        c: float = field()
        d: float = field()
        e: float = field()

    @transform
    def c(a: float) -> float:
        return a + 1

    @transform
    def d(c: float) -> float:
        return c * 2

    @transform
    def e(c: float, d: float) -> float:
        return c + d

    implementation = Implementation(Demo, [c, d, e])
    analyser = implementation.analyser_for(backend)
    analyse = analyser.analyse
    plans: list[BackendCase] = []

    def record(request: EvaluationRequest) -> BackendCase:
        case = analyse(request)
        plans.append(case)
        return case

    monkeypatch.setattr(analyser, "analyse", record)
    supplied = 3.0
    instance = Demo(a=supplied)
    with backend, implementation:
        assert instance.d == pytest.approx(8.0)
        cached = _variables(instance)["c"]
        assert instance.c is cached
        assert instance.e == pytest.approx(12.0)

    assert [[output.port for output in plan.outputs] for plan in plans] == [["c", "d"], ["e"]]
    assert _variables(instance)["a"] is supplied
    assert _variables(instance)["c"] is cached


def test_ports_without_interface_fields_are_retained(backend: Backend) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        d: float = field()

    @transform
    def c__local(a: float) -> float:
        return a + 1

    @transform
    def d(c__local: float) -> float:
        return c__local * 2

    with backend, Implementation(Demo, [c__local, d]):
        instance = Demo(a=3.0)
        assert instance.d == pytest.approx(8.0)

    # Double underscores in an ordinary name must not be guessed back into dots.
    assert _variables(instance)["c__local"] == pytest.approx(4.0)
    assert "c.local" not in _variables(instance)


def test_dotted_child_ports_are_cached_without_exposing_child_internals(backend: Backend) -> None:
    @system
    class Child(metaclass=Interface):
        a: float = field()
        b: float = field()

    @transform
    def hidden(a: float) -> float:
        return a + 1

    @transform
    def b(hidden: float) -> float:
        return hidden * 2

    @system
    class Parent(metaclass=Interface):
        source: float = field()
        result: float = field()
        later: float = field()

    @equation
    def connect(source: float, child: Child) -> bool:
        return source == child.a

    @equation
    def finish(result: float, child: Child) -> bool:
        return result == child.b + 1

    @equation
    def reuse(later: float, child: Child) -> bool:
        return later == child.b * 3

    child = Implementation(Child, [hidden, b])
    implementation = Implementation(Parent, [connect, finish, reuse], internal_implementations={"child": child})
    with backend, implementation:
        instance = Parent(source=3.0)
        assert instance.result == pytest.approx(9.0)
        assert _variables(instance)["child.b"] == pytest.approx(8.0)
        # A later parent request must succeed using the dotted value alone.
        assert implementation.evaluate({"child.b": _variables(instance)["child.b"]}, "later", backend) == pytest.approx(
            24.0
        )
        assert instance.later == pytest.approx(24.0)

    assert not ({"child__a", "child__b", "hidden", "child.hidden"} & _variables(instance).keys())
    assert all(not request.expose_intermediates for request in child._evaluation_cache)  # noqa: SLF001


def test_each_tensor_intermediate_is_lifted_in_its_own_domain_and_unit(backend: Backend) -> None:
    location = Dim("cache_location")
    time = Dim("cache_time")

    @system
    class Demo(metaclass=Interface):
        values: MassTensor = field()
        doubled: MassTensor = field()
        averaged: MassTensor = field()
        total: Mass = field()
        later: MassTensor = field()

    @equation
    def double(values: MassTensor, doubled: MassTensor) -> bool:
        return doubled == values * 2

    @equation
    def average(doubled: MassTensor, averaged: MassTensor) -> bool:
        return averaged == doubled.mean(time)

    @equation
    def sum_locations(averaged: MassTensor, total: Mass) -> bool:
        return total == averaged.sum(location)

    @equation
    def reuse(averaged: MassTensor, total: Mass, later: MassTensor) -> bool:
        return later == averaged + total

    implementation = Implementation(Demo, [double, average, sum_locations, reuse])
    supplied = tensor([[1000.0, 2000.0], [3000.0, 4000.0], [5000.0, 6000.0]], over=(time, location), unit=gram)
    instance = Demo(values=supplied)
    with backend, implementation:
        assert instance.total == 14.0 * kilogram
        assert instance.doubled.domain == domain(time, location)
        assert instance.doubled.unit == kilogram
        np.testing.assert_allclose(instance.doubled.values, [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])
        assert instance.averaged.domain == domain(location)
        assert instance.averaged.unit == kilogram
        np.testing.assert_allclose(instance.averaged.values, [6.0, 8.0])
        np.testing.assert_allclose(instance.later.values, [20.0, 22.0])
    assert instance.values is supplied


def test_scalar_declared_tensor_intermediate_stays_a_tensor(backend: Backend) -> None:
    @system
    class Demo(metaclass=Interface):
        values: MassTensor = field()
        doubled: MassTensor = field()
        result: MassTensor = field()

    @equation
    def double(values: MassTensor, doubled: MassTensor) -> bool:
        return doubled == values * 2

    @equation
    def finish(doubled: MassTensor, result: MassTensor) -> bool:
        return result == doubled * 3

    with backend, Implementation(Demo, [double, finish]):
        instance = Demo(values=tensor(2.0, over=(), unit=kilogram))
        assert instance.result.values == pytest.approx(12.0)
        assert isinstance(instance.doubled, Tensor)
        assert instance.doubled.domain == domain()
        assert instance.doubled.unit == kilogram


def test_scalar_intermediates_use_their_own_unit_declarations(backend: Backend) -> None:
    @system
    class Resistor(metaclass=Interface):
        i: Current = field()
        r: Resistance = field()
        v: Voltage = field()
        p: Power = field()

    @equation
    def ohms_law(i: Current, r: Resistance, v: Voltage) -> bool:
        return v == i * r

    @equation
    def power(i: Current, v: Voltage, p: Power) -> bool:
        return p == i * v

    with backend, Implementation(Resistor, [ohms_law, power]):
        instance = Resistor(i=2.0 * ampere, r=3.0 * ohm)
        assert instance.p == 12.0 * watt
        assert instance.v == 6.0 * volt
        assert instance.v.unit == volt


def test_cached_values_stay_with_the_selected_implementation(backend: Backend) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def b(a: float) -> float:
        return a + 1

    @transform
    def c(b: float) -> float:
        return b * 2

    first = Implementation(Demo, [b, c])

    @transform
    def b(a: float) -> float:
        return a + 10

    second = Implementation(Demo, [b, c])
    instance = Demo(a=3.0)
    with backend, first:
        assert instance.c == pytest.approx(8.0)
    with backend, second:
        assert instance.b == pytest.approx(4.0)
        assert Demo(a=3.0).c == pytest.approx(26.0)


def test_single_value_and_intermediate_compilation_modes_are_separate(backend: Backend) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()

    @transform
    def b(a: float) -> float:
        return a + 1

    implementation = Implementation(Demo, [b])
    inputs: dict[str, object] = {"a": 3.0}
    assert implementation.evaluate(inputs, "b", backend) == pytest.approx(4.0)
    with backend, implementation:
        assert Demo(a=3.0).b == pytest.approx(4.0)
    assert implementation.evaluate(inputs, "b", backend) == pytest.approx(4.0)
    assert {request.expose_intermediates for request in implementation._evaluation_cache} == {False, True}  # noqa: SLF001
    supplied = 7.0
    assert implementation.evaluate({"b": supplied}, "b", backend) is supplied


def test_instance_compilation_respects_coordinate_context(backend: Backend) -> None:
    time = Dim("cache_selection_time")

    @system
    class Demo(metaclass=Interface):
        values: Tensor = field()
        selected: float = field()
        result: float = field()

    @equation
    def select(values: Tensor, selected: float) -> bool:
        return selected == values.sel(time, 2025)

    @transform
    def result(selected: float) -> float:
        return selected * 2

    implementation = Implementation(Demo, [select, result])
    for coordinates, expected in (((2024, 2025), 20.0), ((2025, 2024), 10.0)):
        with backend, implementation, EvaluationContext() as evaluation:
            evaluation.domains.bind(coordinate_axis(time, coordinates))
            instance = Demo(values=tensor([10.0, 20.0], over=time))
            assert instance.result == pytest.approx(expected * 2)
            assert instance.selected == pytest.approx(expected)


@pytest.mark.parametrize("failure_stage", ("execution", "lifting"))
def test_failure_leaves_instance_values_untouched(
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()
        c: float = field()

    @transform
    def b(a: float) -> float:
        return a + 1

    @transform
    def c(b: float) -> float:
        return b * 2

    implementation = Implementation(Demo, [b, c])
    original_lift = implementation_module.lift_output
    lifts = 0

    def fail_execution(**_: object) -> object:
        msg = "Execution failed"
        raise RuntimeError(msg)

    def fail_lifting(value: object, plan: LiftingPlan) -> object:
        nonlocal lifts
        lifts += 1
        if lifts > 1:
            msg = "Lifting failed"
            raise RuntimeError(msg)
        return original_lift(value, plan)

    if failure_stage == "execution":
        chain_case = backend.chain_case

        def failing_case(case: BackendCase) -> BackendCallable:
            callable_ = chain_case(case)
            callable_.callable = fail_execution
            return callable_

        monkeypatch.setattr(backend, "chain_case", failing_case)
    else:
        monkeypatch.setattr(implementation_module, "lift_output", fail_lifting)

    instance = Demo(a=3.0)
    before = _variables(instance).copy()
    with backend, implementation, pytest.raises(RuntimeError, match="failed"):
        _ = instance.c
    assert _variables(instance) == before
    assert _variables(instance)["b"] is MISSING


@pytest.mark.parametrize("invalid_outputs", ({"a": 99.0, "b": 4.0}, {"other": 4.0}))
def test_invalid_evaluation_outputs_do_not_modify_the_instance(
    backend: Backend,
    monkeypatch: pytest.MonkeyPatch,
    invalid_outputs: dict[str, object],
) -> None:
    @system
    class Demo(metaclass=Interface):
        a: float = field()
        b: float = field()

    implementation = Implementation(Demo, [])
    monkeypatch.setattr(implementation, "_evaluate_with_intermediates", lambda **_: invalid_outputs)
    instance = Demo(a=3.0)
    before = _variables(instance).copy()
    with backend, implementation, pytest.raises(RuntimeError, match=r"recomputed|did not produce"):
        _ = instance.b
    assert _variables(instance) == before
