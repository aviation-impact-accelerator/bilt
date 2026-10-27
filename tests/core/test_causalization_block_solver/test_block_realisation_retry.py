import typing

import pytest

from bilt import equation
from bilt._backend._case import BackendCase
from bilt._backend._errors import BiltUnsupportedNodeError
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis import _realize as realization
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock, PlanStep
from bilt.core._structural_analysis._realize import BlockRealizationFailure, BlockRealizationKey, realize_plan
from bilt.core._tensors._errors import BiltSymbolicLoweringError, BiltTensorDomainError
from bilt.symbolic._dimensions import Dim
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import ast
    import collections.abc

    import rustworkx as rx

    from bilt._backend import BackendCallable
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._realize import RealizationFailure


@equation
def total(x: float, y: float, p: float) -> bool:
    return x + y == p


@equation
def difference(x: float, y: float, q: float) -> bool:
    return x - y == q


def _request() -> EvaluationRequest:
    return EvaluationRequest(output="x", input_domains=dict.fromkeys({"p", "q"}, Domain.scalar()))


def _plan(*, swapped: bool = False, domain: Domain | None = None) -> AnalysisPlan:
    operations = (difference, total) if swapped else (total, difference)
    block = PlanBlock(
        steps=tuple(
            PlanStep(operation=operation, output=output)
            for operation, output in zip(operations, ("x", "y"), strict=True)
        ),
        is_cyclic=True,
    )
    request = _request()
    if domain is not None:
        request = EvaluationRequest(output="x", input_domains={"p": domain, "q": Domain.scalar()})
    return AnalysisPlan(request=request, blocks=(block,))


def test_rejected_block_retries_another_matching_without_blacklisting_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    attempts = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        attempts.append(function)
        if len(attempts) == 1:
            message = "first block formulation refused"
            raise BiltUnsupportedNodeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    analyser = solver.CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND)
    with EvaluationContext():
        case = analyser.analyse(_request())
    assert PYTHON_BACKEND.chain_case(case).callable(p=10.0, q=2.0) == pytest.approx(6.0)
    expected_attempts = 2
    assert len(attempts) == expected_attempts
    assert not analyser._failures  # noqa: SLF001 - block rejection must not become a step exclusion


def test_all_block_rejections_terminate_and_do_not_survive_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = []
    error = BiltUnsupportedNodeError("all block formulations refused")

    def refuse(function: ast.FunctionDef) -> typing.Never:
        attempts.append(function)
        raise error

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", refuse)
    analyser = solver.CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND)
    with EvaluationContext():
        for request_number in (1, 2):
            with pytest.raises(BiltExecutionPlanError, match="all block formulations refused") as raised:
                analyser.analyse(_request())
            assert isinstance(raised.value.__cause__, BiltLinearBlockError)
            assert raised.value.__cause__.__cause__ is error
            expected_attempts_per_request = 2
            assert len(attempts) == expected_attempts_per_request * request_number
            assert not analyser._failures  # noqa: SLF001


def test_duplicate_block_attempts_reuse_rejection_but_other_matchings_remain_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PYTHON_BACKEND.transpile_linear_solve
    attempts = []

    def compile_solve(function: ast.FunctionDef) -> BackendCallable:
        attempts.append(function)
        if len(attempts) == 1:
            message = "block refused"
            raise BiltUnsupportedNodeError(message)
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", compile_solve)
    failures: dict[BlockRealizationKey, BlockRealizationFailure] = {}
    with EvaluationContext():
        rejected = realize_plan(_plan(), PYTHON_BACKEND, block_failures=failures)
        repeated = realize_plan(_plan(), PYTHON_BACKEND, block_failures=failures)
        accepted = realize_plan(_plan(swapped=True), PYTHON_BACKEND, block_failures=failures)
    assert isinstance(rejected, BlockRealizationFailure)
    assert repeated is rejected
    assert isinstance(accepted, BackendCase)
    expected_attempts = 2
    assert len(attempts) == expected_attempts
    assert PYTHON_BACKEND.chain_case(accepted).callable(p=10.0, q=2.0) == pytest.approx(6.0)


def test_block_rejection_is_specific_to_its_incoming_domains() -> None:
    failures: dict[BlockRealizationKey, BlockRealizationFailure] = {}
    with EvaluationContext():
        rejected = realize_plan(_plan(domain=Domain((Dim("axis"),))), PYTHON_BACKEND, block_failures=failures)
        accepted = realize_plan(_plan(), PYTHON_BACKEND, block_failures=failures)
    assert isinstance(rejected, BlockRealizationFailure)
    assert "must be registered" in str(rejected.cause)
    assert isinstance(accepted, BackendCase)
    assert PYTHON_BACKEND.chain_case(accepted).callable(p=10.0, q=2.0) == pytest.approx(6.0)


@pytest.mark.parametrize("error_type", (BiltSymbolicLoweringError, BiltTensorDomainError))
def test_lowering_failure_reports_the_entire_block_and_preserves_its_cause(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    error = error_type("unsupported block lowering")

    def refuse(*_: object, **__: object) -> typing.Never:
        raise error

    monkeypatch.setattr(realization, "realize_linear_block", refuse)
    plan = _plan()
    with EvaluationContext():
        failure = realize_plan(plan, PYTHON_BACKEND)
    assert isinstance(failure, BlockRealizationFailure)
    assert failure.failed_block is plan.blocks[0]
    assert failure.cause is error


def test_projection_compilation_failure_is_a_block_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PYTHON_BACKEND.transpile
    error = BiltUnsupportedNodeError("projection refused")

    def compile_function(function: ast.FunctionDef) -> BackendCallable:
        if function.name == "x":
            raise error
        return original(function)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile", compile_function)
    with EvaluationContext():
        failure = realize_plan(_plan(), PYTHON_BACKEND)
    assert isinstance(failure, BlockRealizationFailure)
    assert failure.cause.__cause__ is error


def test_unexpected_programming_errors_are_not_treated_as_rejected_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object, **__: object) -> typing.Never:
        message = "unexpected compiler bug"
        raise RuntimeError(message)

    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", broken)
    analyser = solver.CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND)
    with EvaluationContext(), pytest.raises(RuntimeError, match="unexpected compiler bug"):
        analyser.analyse(_request())


def test_surplus_assignments_do_not_recompile_the_same_rejected_block(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def surplus(x: float, u: float, v: float) -> bool:
        return u + v == x

    original = solver.search.candidate_assignments
    candidates = []
    attempts = []

    def assignments(
        graph: rx.PyGraph,
        failures: collections.abc.Mapping[PlanStep, RealizationFailure],
        *,
        output: str | None,
    ) -> collections.abc.Iterator[dict[str, BiltModelConstituent]]:
        for candidate in original(graph, failures, output=output):
            candidates.append(candidate)
            yield candidate

    def refuse(function: ast.FunctionDef) -> typing.Never:
        attempts.append(function)
        message = "all block formulations refused"
        raise BiltUnsupportedNodeError(message)

    monkeypatch.setattr(solver.search, "candidate_assignments", assignments)
    monkeypatch.setattr(PYTHON_BACKEND, "transpile_linear_solve", refuse)
    analyser = solver.CausalizationStructuralAnalyzer([total, difference, surplus], PYTHON_BACKEND)
    with EvaluationContext(), pytest.raises(BiltExecutionPlanError, match="all block formulations refused"):
        analyser.analyse(_request())
    expected_block_formulations = 2
    assert len(attempts) == expected_block_formulations
    assert len(candidates) > len(attempts)
