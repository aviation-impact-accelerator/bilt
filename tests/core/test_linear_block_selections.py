import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system import _implementation as implementation_module
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.errors import BiltLinearSolveError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import Domain, domain
from bilt.units import Quantity
from bilt.units.dimensions import LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCase
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.symbolic._dimensions import Coordinate, Dim

TIME = dim("time")
SITE = dim("site")
OverTime = typing.Annotated[Tensor, domain(TIME)]
Grid = typing.Annotated[Tensor, domain(TIME, SITE)]
LengthArray = typing.Annotated[Tensor, LENGTH]
LengthOverTime = typing.Annotated[Tensor, LENGTH, domain(TIME)]
LengthScalar = typing.Annotated[Quantity, LENGTH]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@equation
def balance(x: OverTime, y: OverTime, p: Tensor) -> bool:
    return x + y == p


@equation
def select_first(x: OverTime, y: OverTime, q: Tensor) -> bool:
    return x - 0.25 * y.isel(TIME, 0) == q


@system
class Pair(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: Tensor = field()
    q: Tensor = field()


def _evaluate(
    operations: typing.Collection[BiltModelConstituent], values: dict[str, object], output: str, backend: Backend
) -> tuple[np.ndarray, BackendCase]:
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(values.values())
        inputs = {
            name: value.domain if isinstance(value, Tensor) else Domain.scalar() for name, value in values.items()
        }
        case = CausalizationStructuralAnalyzer(operations, backend).analyse(
            EvaluationRequest(output=output, input_domains=inputs)
        )
        aligned = backend.apply_alignment(lower_inputs(values), generate_alignment_plan(values))
        result = backend.chain_case(case).evaluate(aligned)
        return np.asarray(result), case


@pytest.mark.parametrize("index", (0, -1))
@pytest.mark.parametrize("output", ("x", "y"))
def test_literal_position_connects_tensor_coordinates(backend: Backend, index: int, output: str) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.isel(TIME, index) == q

    p = np.array([2.0, 4.0, 6.0])
    q = np.array([0.0, 1.0, 2.0])
    expected_x = q + 0.2 * (p[index] - q[index])
    result, case = _evaluate([balance, second], {"p": tensor(p, over=TIME), "q": tensor(q, over=TIME)}, output, backend)
    np.testing.assert_allclose(result, expected_x if output == "x" else p - expected_x, atol=1e-5)
    assert next(operation for operation in case.intermediates if operation.name == output).output_domain == domain(TIME)


@pytest.mark.parametrize("coordinates", ((10, 20, 30), ("start", "reference", "end"), (0.1, 0.2, 0.3)))
def test_literal_coordinate_uses_the_bound_axis(backend: Backend, coordinates: tuple[Coordinate, ...]) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.sel(TIME, coordinates[1]) == q

    with EvaluationContext() as evaluation:
        evaluation.domains.bind(coordinate_axis(TIME, coordinates))
        result, _ = _evaluate(
            [balance, second],
            {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": tensor([0.0, 1.0, 2.0], over=TIME)},
            "x",
            backend,
        )
    np.testing.assert_allclose(result, [0.6, 1.6, 2.6], atol=1e-5)


@pytest.mark.parametrize("selected_dim", (SITE, TIME), ids=("first-axis", "last-axis"))
def test_selection_respects_named_axes_and_domain_order(backend: Backend, selected_dim: Dim) -> None:
    @equation
    def first(x: Grid, y: Grid, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Grid, y: Grid, q: Tensor) -> bool:
        return x - 0.25 * y.isel(selected_dim, 1) == q

    p = np.arange(6.0).reshape(2, 3) + 2
    q = np.zeros((2, 3))
    axis = 0 if selected_dim is SITE else 1
    result, _ = _evaluate(
        [first, second], {"p": tensor(p, over=(SITE, TIME)), "q": tensor(q, over=(SITE, TIME))}, "x", backend
    )
    np.testing.assert_allclose(result, np.broadcast_to(0.2 * np.take(p, [1], axis=axis), p.shape), atol=1e-5)


def test_selection_outputs_keep_units_through_downstream_reduction(backend: Backend) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: LengthArray = field()
        y: LengthArray = field()
        p: LengthArray = field()
        q: LengthArray = field()
        answer: LengthScalar = field()

    @equation
    def first(x: LengthOverTime, y: LengthOverTime, p: LengthArray) -> bool:
        return x + y == p

    @equation
    def second(x: LengthOverTime, y: LengthOverTime, q: LengthArray) -> bool:
        return x - 0.25 * y.isel(TIME, 0) == q

    @equation
    def finish(answer: LengthScalar, x: LengthArray) -> bool:
        return answer == x.sum(TIME)

    with backend, Implementation(Lengths, [first, second, finish]):
        model = Lengths(
            p=tensor([0.01, 0.02, 0.03], over=TIME, unit=kilometer),
            q=tensor([2.0, 4.0, 6.0], over=TIME, unit=meter),
        )
        x = model.x
        answer = model.answer
    assert x.domain == domain(TIME)
    assert x.unit == meter
    np.testing.assert_allclose(x.values, [3.6, 5.6, 7.6], atol=1e-5)
    assert isinstance(answer, Quantity)
    assert answer.unit == meter
    assert float(answer.value) == pytest.approx(16.8, abs=1e-5)


def test_compiled_selection_accepts_different_axis_lengths(backend: Backend) -> None:
    with backend, Implementation(Pair, [balance, select_first]):
        first = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=tensor([0.0, 1.0, 2.0], over=TIME)).x
        second = Pair(p=tensor([2.0, 4.0], over=TIME), q=tensor([0.0, 1.0], over=TIME)).x
    np.testing.assert_allclose(first.values, [0.4, 1.4, 2.4], atol=1e-5)
    np.testing.assert_allclose(second.values, [0.4, 1.4], atol=1e-5)


def test_missing_coordinate_does_not_poison_later_analysis(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.sel(TIME, "reference") == q

    analyser = CausalizationStructuralAnalyzer([balance, second], backend)
    inputs = dict.fromkeys(("p", "q"), domain(TIME))
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(coordinate_axis(TIME, ("a", "b")))
        with pytest.raises(BiltExecutionPlanError, match="coordinate bindings"):
            analyser.analyse(EvaluationRequest(output="x", input_domains=inputs))
    with EvaluationContext() as evaluation:
        evaluation.domains.bind(coordinate_axis(TIME, ("reference", "b")))
        case = analyser.analyse(EvaluationRequest(output="x", input_domains=inputs))
        result = backend.chain_case(case).callable(p=np.array([2.0, 4.0]), q=np.array([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(result), [0.4, 1.4], atol=1e-5)


def test_selection_requires_the_axis_in_the_unknown_domain(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.isel(SITE, 0) == q

    with pytest.raises(BiltExecutionPlanError, match="Cannot index dimension"):
        _evaluate(
            [balance, second], {"p": tensor([2.0, 4.0], over=TIME), "q": tensor([0.0, 1.0], over=TIME)}, "x", backend
        )


def test_selection_infers_unknown_domains(backend: Backend) -> None:
    @equation
    def first(x: Tensor, y: Tensor, p: Tensor) -> bool:
        return x + y == p

    @equation
    def second(x: Tensor, y: Tensor, q: Tensor) -> bool:
        return x - 0.25 * y.isel(TIME, 0) == q

    result, case = _evaluate(
        [first, second], {"p": tensor([2.0, 4.0], over=TIME), "q": tensor([0.0, 1.0], over=TIME)}, "x", backend
    )
    np.testing.assert_allclose(result, [0.4, 1.4], atol=1e-5)
    assert next(operation for operation in case.intermediates if operation.name == "x").output_domain == domain(TIME)


def test_known_input_selector_remains_unsupported(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor, index: int) -> bool:
        return x - 0.25 * y.isel(TIME, index) == q

    with pytest.raises(BiltExecutionPlanError, match="coupled blocks"):
        _evaluate(
            [balance, second],
            {"p": tensor([2.0, 4.0], over=TIME), "q": tensor([0.0, 1.0], over=TIME), "index": 0},
            "x",
            backend,
        )


def test_unknown_dependent_selector_remains_unsupported(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.isel(TIME, typing.cast("int", x)) == q

    with pytest.raises(BiltExecutionPlanError, match="coupled blocks"):
        _evaluate(
            [balance, second], {"p": tensor([2.0, 4.0], over=TIME), "q": tensor([0.0, 1.0], over=TIME)}, "x", backend
        )


def test_float_position_remains_unsupported(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.isel(TIME, typing.cast("int", 0.0)) == q

    with pytest.raises(BiltExecutionPlanError, match="coupled blocks"):
        _evaluate(
            [balance, second], {"p": tensor([2.0, 4.0], over=TIME), "q": tensor([0.0, 1.0], over=TIME)}, "x", backend
        )


def test_out_of_range_position_does_not_poison_later_values(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.isel(TIME, 1) == q

    with backend, Implementation(Pair, [balance, second]):
        with pytest.raises((IndexError, BiltLinearSolveError, RuntimeError), match=r"bounds|finite"):
            np.asarray(Pair(p=tensor([2.0], over=TIME), q=tensor([0.0], over=TIME)).x.values)
        result = Pair(p=tensor([2.0, 4.0], over=TIME), q=tensor([0.0, 1.0], over=TIME)).x
    np.testing.assert_allclose(result.values, [0.6, 1.6], atol=1e-5)


def test_jax_selection_solve_remains_differentiable() -> None:
    _, case = _evaluate(
        [balance, select_first],
        {"p": tensor([2.0, 4.0], over=TIME), "q": tensor([0.0, 1.0], over=TIME)},
        "x",
        JAX_BACKEND,
    )
    compiled = JAX_BACKEND.chain_case(case)
    derivative = jax.jacrev(lambda p: compiled.callable(p=p, q=jnp.array([0.0, 1.0])))(jnp.array([2.0, 4.0]))
    np.testing.assert_allclose(derivative, [[0.2, 0.0], [0.2, 0.0]], atol=1e-5)


def test_coordinate_selection_cache_tracks_changed_coordinate_order(backend: Backend) -> None:
    @equation
    def second(x: OverTime, y: OverTime, q: Tensor) -> bool:
        return x - 0.25 * y.sel(TIME, "reference") == q

    implementation = Implementation(Pair, [balance, second])
    results = []
    for coordinates in (("reference", "other"), ("other", "reference")):
        with EvaluationContext() as evaluation, backend, implementation:
            evaluation.domains.bind(coordinate_axis(TIME, coordinates))
            result = Pair(p=tensor([2.0, 4.0], over=TIME), q=tensor([0.0, 1.0], over=TIME)).x
            results.append(np.asarray(result.values))
    np.testing.assert_allclose(results[0], [0.4, 1.4], atol=1e-5)
    np.testing.assert_allclose(results[1], [0.6, 1.6], atol=1e-5)
