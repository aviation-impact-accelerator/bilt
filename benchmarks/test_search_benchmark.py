import typing

import pytest
import structural_search as search

from bilt.core._errors import BiltAnalysisError, BiltNoActiveEvaluationError
from bilt.core._evaluation_context import current_evaluation
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._system import _implementation as implementation_module
from bilt.errors import BiltIsolationError, BiltUnsupportedNodeError

if typing.TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def prepare_trial(request: pytest.FixtureRequest) -> Callable[[search.SearchCase], search.SearchTrial]:
    def prepare(case: search.SearchCase) -> search.SearchTrial:
        trial = search.SearchTrial.prepare(case)
        request.addfinalizer(trial.close)
        return trial

    return prepare


@pytest.mark.parametrize("solver", ("exhaustive", "scc"))
def test_trials_use_fresh_models_and_contexts(
    prepare_trial: Callable[[search.SearchCase], search.SearchTrial],
    solver: str,
) -> None:
    case = search.SearchCase(solver=solver, layout="siblings", coupled=False, depth=1, rectangular=False)
    original_factory = implementation_module.ExhaustiveStructuralAnalyser
    first = prepare_trial(case)
    first_context = current_evaluation()
    first.analyse()
    assert first.check()["value"] == search.expected_value(coupled=False, layout="siblings", depth=1)
    first.close()
    second = prepare_trial(case)
    assert second.implementation is not first.implementation
    assert current_evaluation() is not first_context
    second.analyse()
    assert second.check()["value"] == first.check()["value"]
    second.close()
    assert implementation_module.ExhaustiveStructuralAnalyser is original_factory
    with pytest.raises(BiltNoActiveEvaluationError):
        current_evaluation()


def test_trial_cannot_time_a_cached_second_analysis(
    prepare_trial: Callable[[search.SearchCase], search.SearchTrial],
) -> None:
    trial = prepare_trial(search.SearchCase(solver="scc", layout="links", coupled=False, depth=0, rectangular=False))
    trial.analyse()
    with pytest.raises(RuntimeError, match="analysed once"):
        trial.analyse()


def test_wrong_numeric_answer_is_rejected(
    prepare_trial: Callable[[search.SearchCase], search.SearchTrial],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial = prepare_trial(search.SearchCase(solver="scc", layout="links", coupled=False, depth=0, rectangular=False))
    trial.analyse()
    monkeypatch.setattr(search, "expected_value", lambda **_: -1.0)
    with pytest.raises(AssertionError, match="Expected -1"):
        trial.check()


@pytest.mark.parametrize("error_type", (BiltOverconstrainedError, BiltUnderconstrainedError))
def test_wrong_structural_diagnosis_is_not_a_coupled_pass(
    prepare_trial: Callable[[search.SearchCase], search.SearchTrial],
    error_type: type[BiltAnalysisError],
) -> None:
    trial = prepare_trial(search.SearchCase(solver="scc", layout="links", coupled=True, depth=0, rectangular=False))
    trial.result = error_type("Wrong structural diagnosis")
    with pytest.raises(AssertionError, match="Expected unsupported coupled execution"):
        trial.check()


def test_analysis_failure_is_not_an_acyclic_pass(
    prepare_trial: Callable[[search.SearchCase], search.SearchTrial],
) -> None:
    trial = prepare_trial(search.SearchCase(solver="scc", layout="links", coupled=False, depth=0, rectangular=False))
    trial.result = BiltExecutionPlanError("No route")
    with pytest.raises(AssertionError, match="Expected a successful plan"):
        trial.check()


@pytest.mark.parametrize("solver", ("exhaustive", "scc"))
@pytest.mark.parametrize("wrapped", (False, True))
@pytest.mark.parametrize("error_type", (BiltIsolationError, BiltUnsupportedNodeError))
def test_realisation_error_is_not_a_coupled_pass(
    solver: str,
    error_type: type[Exception],
    *,
    wrapped: bool,
) -> None:
    error: Exception = error_type("Unexpected realisation failure")
    if wrapped:
        outer_type = BiltExecutionPlanError if solver == "scc" else BiltAnalysisError
        outer = outer_type(
            "Cannot compute x from available inputs {'a', 'b'}. Execution of coupled blocks is not yet supported"
        )
        outer.__cause__ = error
        error = outer
    with pytest.raises(AssertionError, match="Expected unsupported coupled execution"):
        search.check_coupled_rejection(error, solver=solver, output="x")


def test_unexplained_execution_failure_is_not_a_coupled_pass() -> None:
    with pytest.raises(AssertionError, match="Expected unsupported coupled execution"):
        search.check_coupled_rejection(BiltExecutionPlanError("No supported route"), solver="scc", output="x")


def test_coupled_rejection_preserves_legacy_diagnostic_limit() -> None:
    error = BiltAnalysisError("Cannot compute x from available inputs {'a', 'b'}.")
    assert (
        search.check_coupled_rejection(error, solver="exhaustive", output="x")["rejection_reason"] == "no_acyclic_plan"
    )


def test_wrapped_coupled_diagnostic_is_accepted() -> None:
    error = BiltExecutionPlanError("Cannot compute x from available inputs {'a', 'b'}.")
    error.__cause__ = BiltExecutionPlanError("Execution of coupled blocks is not yet supported by this analyser.")
    assert (
        search.check_coupled_rejection(error, solver="scc", output="x")["rejection_reason"]
        == "unsupported_coupled_execution"
    )


def test_cyclic_exception_chain_does_not_hang() -> None:
    error = BiltExecutionPlanError("Execution of coupled blocks is not yet supported")
    error.__cause__ = error
    with pytest.raises(AssertionError, match="Expected unsupported coupled execution"):
        search.check_coupled_rejection(error, solver="scc", output="x")


def test_numerical_success_is_not_a_coupled_pass(
    prepare_trial: Callable[[search.SearchCase], search.SearchTrial],
) -> None:
    trial = prepare_trial(search.SearchCase(solver="scc", layout="links", coupled=True, depth=0, rectangular=False))
    trial.result = search.BackendCase("x", (), [], "x")
    with pytest.raises(AssertionError, match="Expected unsupported coupled execution"):
        trial.check()


def test_setup_failure_restores_solver_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    original_factory = implementation_module.ExhaustiveStructuralAnalyser

    def broken(**_: object) -> typing.Never:
        msg = "Model setup failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(search, "build_model", broken)
    with pytest.raises(RuntimeError, match="Model setup failed"):
        search.SearchTrial.prepare(
            search.SearchCase(solver="scc", layout="links", coupled=False, depth=0, rectangular=False)
        )
    assert implementation_module.ExhaustiveStructuralAnalyser is original_factory
    with pytest.raises(BiltNoActiveEvaluationError):
        current_evaluation()


@pytest.mark.parametrize("solver", ("exhaustive", "scc"))
def test_counter_trial_checks_answer_and_restores_patches(solver: str) -> None:
    original_factory = implementation_module.ExhaustiveStructuralAnalyser
    original_matching = search.scc_search.find_matching
    original_plan = search.scc_planning.plan_steps
    original_settlement = search.scc_planning.settle_assignments
    details = search.count_search(
        search.SearchCase(solver=solver, layout="links", coupled=False, depth=1, rectangular=True)
    )
    assert details["status"] == "success"
    assert details["value"] == search.expected_value(coupled=False, layout="links", depth=1)
    assert details["counts"]
    if solver == "scc":
        counts = typing.cast("dict[str, int]", details["counts"])
        assert counts["execution_matching_calls"] > 0
        assert counts["settlement_calls"] > 0
        assert counts["candidate_checks"] > 0
    assert implementation_module.ExhaustiveStructuralAnalyser is original_factory
    assert search.scc_search.find_matching is original_matching
    assert search.scc_planning.plan_steps is original_plan
    assert search.scc_planning.settle_assignments is original_settlement
    with pytest.raises(BiltNoActiveEvaluationError):
        current_evaluation()


def test_counter_failure_restores_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    original_factory = implementation_module.ExhaustiveStructuralAnalyser
    original_matching = search.scc_search.find_matching
    original_plan = search.scc_planning.plan_steps
    original_settlement = search.scc_planning.settle_assignments
    monkeypatch.setattr(search, "expected_value", lambda **_: -1.0)
    with pytest.raises(AssertionError, match="Expected -1"):
        search.count_search(search.SearchCase(solver="scc", layout="links", coupled=False, depth=0, rectangular=False))
    assert implementation_module.ExhaustiveStructuralAnalyser is original_factory
    assert search.scc_search.find_matching is original_matching
    assert search.scc_planning.plan_steps is original_plan
    assert search.scc_planning.settle_assignments is original_settlement
    with pytest.raises(BiltNoActiveEvaluationError):
        current_evaluation()


pytest_plugins = ("pytester",)


@pytest.mark.parametrize("failure_mode", ("disabled", "middle-round"))
def test_pytest_runner_checks_disabled_and_intermediate_rounds(
    pytester: pytest.Pytester,
    failure_mode: str,
) -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[2]
    pytester.makeini(f"[pytest]\npythonpath = {root / 'src'} {root / 'benchmarks'}\n")
    checks = """
import structural_search as search

@pytest.fixture(autouse=True)
def wrong_reference(monkeypatch):
    original = search.expected_value
    calls = 0
    def reference(**kwargs):
        nonlocal calls
        calls += 1
        if MODE == "disabled" or calls == 2:
            return -1.0
        return original(**kwargs)
    monkeypatch.setattr(search, "expected_value", reference)
""".replace("MODE", repr(failure_mode))
    pytester.makeconftest((root / "benchmarks/conftest.py").read_text() + checks)
    pytester.makepyfile(test_nested_equation_search=(root / "benchmarks/test_nested_equation_search.py").read_text())
    options = [
        "--search-extended",
        "--search-rounds=3",
        "--timeout=30",
        "-k",
        "scc and depth0 and acyclic and links and square",
    ]
    if failure_mode == "disabled":
        options.append("--benchmark-disable")
    result = pytester.runpytest_subprocess(*options)
    result.assert_outcomes(failed=1, deselected=71)
    result.stdout.fnmatch_lines(["*Expected -1.0, received 6.0*"])
