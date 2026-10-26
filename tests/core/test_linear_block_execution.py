import dataclasses
import itertools
import types
import typing

import jax
import numpy as np
import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.errors import BiltAnalysisError, BiltLinearSolveError
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain
from bilt.units import Quantity
from bilt.units.dimensions import LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCase
    from bilt.core._model_constituent import BiltModelConstituent

Length = typing.Annotated[Quantity, LENGTH]


@system
class CoupledLengths(metaclass=Interface):
    x: Length = field()
    y: Length = field()
    p: Length = field()
    q: Length = field()


@equation
def length_total(x: Length, y: Length, p: Length) -> bool:
    return x + y == p


@equation
def length_difference(x: Length, y: Length, q: Length) -> bool:
    return x - y == q


@equation
def total(x: float, y: float, p: float) -> bool:
    return x + y == p


@equation
def difference(x: float, y: float, q: float) -> bool:
    return x - y == q


@transform
def p(raw: float) -> float:
    return 2 * raw


@transform
def answer(x: float, y: float) -> float:
    return x * y


@equation
def parameterised(x: float, y: float, p: float, a: float) -> bool:
    return a * x + y == p


def _case(
    operations: typing.Collection[BiltModelConstituent],
    inputs: dict[str, Domain],
    output: str,
    backend: Backend,
) -> BackendCase:
    analyser = CausalizationStructuralAnalyzer(operations, backend)
    with EvaluationContext():
        return analyser.analyse(EvaluationRequest(output=output, input_domains=inputs))


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize("output", ("x", "y", "answer"))
def test_linear_block_exposes_every_output_to_downstream_steps(backend: Backend, output: str) -> None:
    case = _case([total, difference, p, answer], dict.fromkeys({"raw", "q"}, Domain.scalar()), output, backend)
    compiled = backend.chain_case(case)

    expected = {"x": 6.0, "y": 4.0, "answer": 24.0}[output]
    assert float(compiled.callable(raw=5.0, q=2.0)) == pytest.approx(expected)
    assert all(
        operation.output_domain == Domain.scalar()
        for operation in case.intermediates
        if not operation.name.startswith("__bilt_linear")
    )


def test_one_runtime_solve_per_block_even_when_both_outputs_are_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = typing.cast("typing.Callable[..., object]", np.linalg.solve)

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", counted)
    case = _case([total, difference, answer], dict.fromkeys({"p", "q"}, Domain.scalar()), "answer", PYTHON_BACKEND)
    compiled = PYTHON_BACKEND.chain_case(case)
    assert calls == 0

    expected_first, expected_second = 24.0, 8.0
    assert compiled.callable(p=10.0, q=2.0) == pytest.approx(expected_first)
    assert calls == 1
    assert compiled.callable(p=6.0, q=2.0) == pytest.approx(expected_second)
    expected_calls = 2
    assert calls == expected_calls


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_coefficients_are_recomputed_for_each_evaluation(backend: Backend) -> None:
    case = _case([parameterised, difference], dict.fromkeys({"p", "q", "a"}, Domain.scalar()), "x", backend)
    compiled = backend.chain_case(case)

    assert float(compiled.callable(p=10.0, q=2.0, a=1.0)) == pytest.approx(6.0)
    assert float(compiled.callable(p=10.0, q=2.0, a=3.0)) == pytest.approx(3.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_complex_coefficients_are_not_silently_cast_to_real(backend: Backend) -> None:
    case = _case([parameterised, difference], dict.fromkeys({"p", "q", "a"}, Domain.scalar()), "x", backend)

    actual = backend.chain_case(case).callable(p=10.0, q=2.0, a=1j)

    assert complex(actual) == pytest.approx(6 - 6j)


def test_jax_case_can_be_nested_in_jit() -> None:
    case = _case([total, difference], dict.fromkeys({"p", "q"}, Domain.scalar()), "x", JAX_BACKEND)
    compiled = jax.jit(JAX_BACKEND.chain_case(case).callable)

    assert float(compiled(p=10.0, q=2.0)) == pytest.approx(6.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_constant_coupled_system_needs_no_inputs(backend: Backend) -> None:
    supplied_sum, supplied_difference = 10, 2

    @equation
    def first(x: float, y: float) -> bool:
        return x + y == supplied_sum

    @equation
    def second(x: float, y: float) -> bool:
        return x - y == supplied_difference

    case = _case([first, second], {}, "x", backend)

    assert float(backend.chain_case(case).callable()) == pytest.approx(6.0)


@pytest.mark.parametrize("order", tuple(itertools.permutations(range(2))))
def test_declaration_order_does_not_change_the_solution(order: tuple[int, ...]) -> None:
    operations = (total, difference)
    case = _case(
        [operations[index] for index in order], dict.fromkeys({"p", "q"}, Domain.scalar()), "x", PYTHON_BACKEND
    )

    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_singular_runtime_values_do_not_poison_the_compiled_case(backend: Backend) -> None:
    case = _case([parameterised, difference], dict.fromkeys({"p", "q", "a"}, Domain.scalar()), "x", backend)
    compiled = backend.chain_case(case)

    # JAX wraps exceptions from runtime callbacks in its own runtime error.
    with pytest.raises((BiltLinearSolveError, RuntimeError), match="singular"):
        float(compiled.callable(p=10.0, q=2.0, a=-1.0))
    assert float(compiled.callable(p=10.0, q=2.0, a=1.0)) == pytest.approx(6.0)


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_nonfinite_inputs_are_reported(bad: float) -> None:
    case = _case([total, difference], dict.fromkeys({"p", "q"}, Domain.scalar()), "x", PYTHON_BACKEND)

    with pytest.raises(BiltLinearSolveError, match="finite"):
        PYTHON_BACKEND.chain_case(case).callable(p=bad, q=2.0)


def test_nonscalar_domains_are_rejected_without_permanent_assignment_failures() -> None:
    analyser = CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND)
    axis = Dim("axis")
    with EvaluationContext():
        with pytest.raises(BiltExecutionPlanError, match="require scalar coefficients"):
            analyser.analyse(EvaluationRequest(output="x", input_domains={"p": Domain((axis,)), "q": Domain.scalar()}))
        case = analyser.analyse(EvaluationRequest(output="x", input_domains=dict.fromkeys({"p", "q"}, Domain.scalar())))

    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)


def test_unsupported_backend_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    bindings = dataclasses.replace(PYTHON_BACKEND.intrinsic_bindings, linear_solve=None)
    monkeypatch.setattr(PYTHON_BACKEND, "intrinsic_bindings", bindings)

    with pytest.raises(BiltExecutionPlanError, match="backend does not support scalar linear blocks"):
        _case([total, difference], dict.fromkeys({"p", "q"}, Domain.scalar()), "x", PYTHON_BACKEND)


def test_nonlinear_coupling_retains_fast_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def nonlinear(x: float, y: float, p: float) -> bool:
        return x * y == p

    def unexpected(*_: object, **__: object) -> typing.Never:
        message = "Execution matching must not run for this unsupported coupled block."
        raise AssertionError(message)

    monkeypatch.setattr(search, "find_matching", unexpected)
    with pytest.raises(BiltAnalysisError, match="coupled blocks"):
        _case([nonlinear, difference], dict.fromkeys({"p", "q"}, Domain.scalar()), "x", PYTHON_BACKEND)


def test_qualified_outputs_are_mangled_for_backend_execution() -> None:
    @system
    class Pair(metaclass=Interface):
        x: float = field()
        y: float = field()

    @equation
    def first(pair: Pair, p: float) -> bool:
        return pair.x + pair.y == p

    @equation
    def second(pair: Pair, q: float) -> bool:
        return pair.x - pair.y == q

    case = _case([first, second], dict.fromkeys({"p", "q"}, Domain.scalar()), "pair.x", PYTHON_BACKEND)

    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_two_linear_blocks_propagate_domains_and_values(backend: Backend) -> None:
    @equation
    def next_total(u: float, v: float, x: float) -> bool:
        return u + v == x

    @equation
    def next_difference(u: float, v: float, y: float) -> bool:
        return u - v == y

    case = _case(
        [total, difference, next_total, next_difference], dict.fromkeys({"p", "q"}, Domain.scalar()), "u", backend
    )

    assert float(backend.chain_case(case).callable(p=10.0, q=2.0)) == pytest.approx(5.0)
    names = [operation.name for operation in case.intermediates if operation.name.startswith("__bilt_linear")]
    expected_solves = 2
    assert len(set(names)) == expected_solves


def test_rectangular_surplus_does_not_prevent_a_local_linear_solution() -> None:
    @equation
    def surplus(u: float, v: float, x: float) -> bool:
        return u + v == x

    case = _case([total, difference, surplus], dict.fromkeys({"p", "q"}, Domain.scalar()), "x", PYTHON_BACKEND)

    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_quantities_are_solved_in_coherent_units(backend: Backend, monkeypatch: pytest.MonkeyPatch) -> None:
    from bilt.core._system import _implementation as implementation_module  # noqa: PLC0415

    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)
    with backend, Implementation(CoupledLengths, [length_total, length_difference]):
        value = CoupledLengths(p=0.01 * kilometer, q=2.0 * meter).x

    assert isinstance(value, Quantity)
    assert value.unit == meter
    assert float(value.value) == pytest.approx(6.0)


@pytest.mark.parametrize("supports_linear_solve", (False, True))
def test_linear_solve_capability_does_not_require_python_intrinsic_bindings(supports_linear_solve: bool) -> None:
    # A backend may supply the compilation capability without exposing the
    # in-process Python implementation's intrinsic registry.
    backend = types.SimpleNamespace(transpile=PYTHON_BACKEND.transpile)
    if supports_linear_solve:
        backend.transpile_linear_solve = PYTHON_BACKEND.transpile_linear_solve
    selected = typing.cast("Backend", backend)
    inputs = dict.fromkeys({"p", "q"}, Domain.scalar())
    if not supports_linear_solve:
        with pytest.raises(BiltExecutionPlanError, match="backend does not support scalar linear blocks"):
            _case([total, difference], inputs, "x", selected)
        return

    case = _case([total, difference], inputs, "x", selected)
    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)
