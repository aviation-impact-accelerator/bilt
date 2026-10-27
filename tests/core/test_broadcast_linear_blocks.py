import json
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
from bilt.core._system import _implementation as implementation_module
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.errors import BiltLinearSolveError
from bilt.symbolic._dimensions import coordinate_axis
from bilt.symbolic._domains import Domain, domain
from bilt.units.dimensions import LENGTH
from bilt.units.units import kilometer, meter

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCase
    from bilt.core._model_constituent import BiltModelConstituent


TIME = dim("time")
SITE = dim("site")
LengthArray = typing.Annotated[Tensor, LENGTH]


@pytest.fixture(autouse=True)
def _use_scc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)


@pytest.fixture(params=(PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    return request.param


@equation
def total(x: float, y: float, p: float, a: float) -> bool:
    return a * x + y == p


@equation
def difference(x: float, y: float, q: float) -> bool:
    return x - y == q


@system
class Pair(metaclass=Interface):
    x: Tensor = field()
    y: Tensor = field()
    p: Tensor = field()
    q: float = field()
    a: float | Tensor = field()


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


@pytest.mark.parametrize("output", ("x", "y"))
def test_pointwise_block_broadcasts_scalar_coefficients(backend: Backend, output: str) -> None:
    with backend, Implementation(Pair, [total, difference]):
        result = getattr(Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=0.0, a=1.0), output)
    assert isinstance(result, Tensor)
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [1, 2, 3], rtol=1e-5)


def test_pointwise_block_aligns_independent_and_reordered_dimensions(backend: Backend) -> None:
    p = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    a = np.array([1.0, 2.0, 3.0])
    q = np.array([2.0, 4.0])
    result, case = _evaluate(
        [total, difference],
        {"p": tensor(p.T, over=(TIME, SITE)), "a": tensor(a, over=TIME), "q": tensor(q, over=SITE)},
        "x",
        backend,
    )
    np.testing.assert_allclose(result, (p.T + q[None, :]) / (a[:, None] + 1), rtol=1e-5)
    output = next(step for step in case.intermediates if step.name == "x")
    assert output.output_domain == domain(TIME, SITE)


def test_pointwise_block_uses_one_batched_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.linalg.solve
    shapes = []

    def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        shapes.append(a.shape)
        return original(a, b)

    monkeypatch.setattr(np.linalg, "solve", solve)
    _evaluate([total, difference], {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 0.0, "a": 1.0}, "x", PYTHON_BACKEND)
    assert shapes == [(3, 2, 2)]


def test_tensor_output_domains_reach_downstream_reductions(backend: Backend) -> None:
    @equation
    def reduce(answer: float, x: Tensor, y: Tensor) -> bool:
        return answer == x.sum(TIME) + y.sum(TIME)

    result, case = _evaluate(
        [total, difference, reduce], {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 0.0, "a": 1.0}, "answer", backend
    )
    np.testing.assert_allclose(result, [12], rtol=1e-5)
    assert case.intermediates[-1].output_domain == Domain.scalar()


def test_singular_batch_does_not_poison_later_values(backend: Backend) -> None:
    implementation = Implementation(Pair, [total, difference])
    with backend, implementation:

        def evaluate_bad() -> np.ndarray:
            result = Pair(p=tensor([2.0, 4.0], over=TIME), q=0.0, a=tensor([-1.0, 1.0], over=TIME)).x
            return np.asarray(result.values)

        with pytest.raises((BiltLinearSolveError, RuntimeError), match="singular"):
            evaluate_bad()
        result = Pair(p=tensor([2.0, 4.0], over=TIME), q=0.0, a=tensor([1.0, 1.0], over=TIME)).x
    np.testing.assert_allclose(result.values, [1, 2], rtol=1e-5)


def test_tensor_units_are_lowered_and_lifted_in_coherent_units(backend: Backend) -> None:
    @system
    class Lengths(metaclass=Interface):
        x: LengthArray = field()
        y: LengthArray = field()
        p: LengthArray = field()
        q: LengthArray = field()

    @equation
    def first(x: LengthArray, y: LengthArray, p: LengthArray) -> bool:
        return x + y == p

    @equation
    def second(x: LengthArray, y: LengthArray, q: LengthArray) -> bool:
        return x - y == q

    with backend, Implementation(Lengths, [first, second]):
        result = Lengths(
            p=tensor([0.01, 0.02], over=TIME, unit=kilometer), q=tensor([2.0, 2.0], over=TIME, unit=meter)
        ).x
    assert result.unit == meter
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [6, 11], rtol=1e-5)


def test_unrelated_context_dimensions_are_not_added_to_the_output(backend: Backend) -> None:
    with EvaluationContext() as evaluation, backend, Implementation(Pair, [total, difference]):
        evaluation.domains.bind(coordinate_axis(SITE, ("a", "b")))
        result = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=0.0, a=1.0).x
    assert result.domain == domain(TIME)
    np.testing.assert_allclose(result.values, [1, 2, 3], rtol=1e-5)


def test_complex_tensor_coefficients_are_preserved(backend: Backend) -> None:
    p = np.array([2.0, 4.0])
    a = np.array([1j, 2j])
    result, _ = _evaluate(
        [total, difference], {"p": tensor(p, over=TIME), "q": 0.0, "a": tensor(a, over=TIME)}, "x", backend
    )
    np.testing.assert_allclose(result, p / (a + 1), rtol=1e-5)


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_nonfinite_tensor_data_is_reported(backend: Backend, bad: float) -> None:
    with pytest.raises((BiltLinearSolveError, RuntimeError), match="finite"):
        _evaluate([total, difference], {"p": tensor([bad, 4.0], over=TIME), "q": 0.0, "a": 1.0}, "x", backend)


def test_empty_pointwise_batch_preserves_its_domain(backend: Backend) -> None:
    with backend, Implementation(Pair, [total, difference]):
        result = Pair(p=tensor([], over=TIME), q=0.0, a=1.0).x
    assert result.domain == domain(TIME)
    assert result.values.shape == (0,)


def test_jax_batched_solve_remains_differentiable() -> None:
    _, case = _evaluate([total, difference], {"p": tensor([2.0, 4.0], over=TIME), "q": 0.0, "a": 1.0}, "x", JAX_BACKEND)
    compiled = JAX_BACKEND.chain_case(case)
    jacobian = jax.jacrev(lambda p: compiled.callable(p=p, q=0.0, a=1.0))(jnp.array([2.0, 4.0]))
    np.testing.assert_allclose(jacobian, 0.5 * np.eye(2), rtol=1e-5)


def test_scalar_solve_retains_a_serializable_scalar_result() -> None:
    with EvaluationContext():
        case = CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND).analyse(
            EvaluationRequest(output="x", input_domains=dict.fromkeys({"p", "q", "a"}, Domain.scalar()))
        )
    result = PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0, a=1.0)
    assert np.isscalar(result)
    assert json.loads(json.dumps(result)) == pytest.approx(6.0)


def test_cached_batched_solve_accepts_different_dimension_extents(backend: Backend) -> None:
    with backend, Implementation(Pair, [total, difference]):
        first = Pair(p=tensor([2.0, 4.0, 6.0], over=TIME), q=0.0, a=1.0).x
        second = Pair(p=tensor([2.0, 4.0], over=TIME), q=0.0, a=1.0).x
    np.testing.assert_allclose(first.values, [1, 2, 3], rtol=1e-5)
    np.testing.assert_allclose(second.values, [1, 2], rtol=1e-5)


def test_two_batched_blocks_propagate_domains(backend: Backend) -> None:
    @equation
    def next_total(u: float, v: float, x: float) -> bool:
        return u + v == x

    @equation
    def next_difference(u: float, v: float, y: float) -> bool:
        return u - v == y

    result, case = _evaluate(
        [total, difference, next_total, next_difference],
        {"p": tensor([2.0, 4.0, 6.0], over=TIME), "q": 0.0, "a": 1.0},
        "u",
        backend,
    )
    np.testing.assert_allclose(result, [1, 2, 3], rtol=1e-5)
    projections = [operation for operation in case.intermediates if operation.output_domain is not None]
    assert all(operation.output_domain == domain(TIME) for operation in projections)


def test_reduction_of_known_inputs_can_supply_scalar_coefficients(backend: Backend) -> None:
    @equation
    def weighted(x: float, y: float, weights: Tensor, p: float) -> bool:
        return weights.sum(TIME) * x + y == p

    result, case = _evaluate(
        [weighted, difference], {"weights": tensor([1.0, 2.0], over=TIME), "p": 10.0, "q": 2.0}, "x", backend
    )
    np.testing.assert_allclose(result, [3], rtol=1e-5)
    assert next(operation for operation in case.intermediates if operation.name == "x").output_domain == Domain.scalar()
