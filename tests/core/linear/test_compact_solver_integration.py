"""End-to-end integration of compact matching with existing linear execution."""

import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._compact_execution import (
    MAX_DENSE_UNKNOWNS,
    try_compact_linear_execution,
)
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan
from bilt.symbolic._domains import domain

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

SOURCE = dim("source")
TARGET = dim("target")

OverSource = typing.Annotated[Tensor, domain(SOURCE)]


@system
class InfluenceSystem(metaclass=Interface):
    x: Tensor = field()
    coefficients: Tensor = field()
    rhs: Tensor = field()


@equation
def influence(x: OverSource, coefficients: Tensor, rhs: Tensor) -> bool:
    return (coefficients * x).sum(SOURCE) == rhs


def _compute(backend: Backend, matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    values: dict[str, object] = {
        "coefficients": tensor(matrix, over=(TARGET, SOURCE)),
        "rhs": tensor(rhs, over=TARGET),
    }
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(values.values())
        case = CausalizationStructuralAnalyzer((influence,), backend).analyse(
            EvaluationRequest("x", {name: value.domain for name, value in values.items() if isinstance(value, Tensor)})
        )
        aligned = backend.apply_alignment(lower_inputs(values), generate_alignment_plan(values))
        actual = backend.chain_case(case).evaluate(aligned)
        return np.asarray(actual).reshape(-1), tuple(step.name for step in case.intermediates)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize("size", (2, 4, 8))
def test_compact_matching_runs_the_existing_linear_solver(backend: Backend, size: int) -> None:
    matrix = np.eye(size) * 3 + np.ones((size, size)) * 0.25
    answer = np.linspace(1.0, 2.0, size)
    result, intermediates = _compute(backend, matrix, matrix @ answer)

    np.testing.assert_allclose(result, answer, atol=1e-5)
    assert intermediates[0] == "__bilt_compact_linear_solution"
    assert intermediates[-1] == "x"


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_public_api_keeps_source_domain_for_inverse_solution(backend: Backend) -> None:
    coefficients = np.array([[2.0, 1.0], [1.0, 3.0]])
    expected = np.array([3.0, 4.0])
    with backend, Implementation(InfluenceSystem, [influence]):
        result = InfluenceSystem(
            coefficients=tensor(coefficients, over=(TARGET, SOURCE)),
            rhs=tensor(coefficients @ expected, over=TARGET),
        ).x

    assert result.domain == domain(SOURCE)
    np.testing.assert_allclose(result.values, expected, atol=1e-5)


def test_rectangular_operator_does_not_use_compact_execution() -> None:
    coefficients = np.ones((2, 3))
    rhs = np.array([1.0, 2.0])
    with pytest.raises(BiltExecutionPlanError):
        _compute(PYTHON_BACKEND, coefficients, rhs)


def test_existing_direct_tensor_equations_do_not_use_compact_joint_solver() -> None:
    @equation
    def direct(x: Tensor, rhs: Tensor) -> bool:
        return x == rhs

    values: dict[str, object] = {"rhs": tensor([1.0, 2.0, 3.0], over=SOURCE)}
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(values.values())
        case = CausalizationStructuralAnalyzer((direct,), PYTHON_BACKEND).analyse(
            EvaluationRequest("x", {"rhs": domain(SOURCE)})
        )
        assert all(step.name != "__bilt_compact_linear_solution" for step in case.intermediates)
        output = PYTHON_BACKEND.chain_case(case).evaluate(
            PYTHON_BACKEND.apply_alignment(lower_inputs(values), generate_alignment_plan(values))
        )
        np.testing.assert_allclose(np.asarray(output).reshape(-1), [1.0, 2.0, 3.0])


def test_large_compact_match_does_not_compile_dense_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large compact graphs must not imply an affordable dense numerical solve."""
    from bilt.core._structural_analysis._causalization_block_solver import _compact_execution

    size = MAX_DENSE_UNKNOWNS + 1
    values = {
        "coefficients": tensor(np.ones((size, size)), over=(TARGET, SOURCE)),
        "rhs": tensor(np.ones(size), over=TARGET),
    }

    def unexpected_dense_solve(*args: object, **kwargs: object) -> None:
        pytest.fail("The compact path attempted to compile an oversized dense solve")

    monkeypatch.setattr(_compact_execution, "realize_linear_block", unexpected_dense_solve)
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_all(values.values())
        analyzer = CausalizationStructuralAnalyzer((influence,), PYTHON_BACKEND)
        request = EvaluationRequest("x", {name: value.domain for name, value in values.items()})
        assert try_compact_linear_execution(analyzer.full_incidence_graph, request, PYTHON_BACKEND) is None
