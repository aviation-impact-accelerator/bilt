import typing
import pytest

from benchmark_trials import benchmark_trials
from recovery_models import RecoveryTrial
from structural_search import instrument

if typing.TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    depths = (0, 1, 2) if metafunc.config.getoption("search_large") else (1,)
    cases = tuple(
        (solver, mode, depth)
        for mode in ("control", "isolation", "backend")
        for depth in depths
        for solver in ("exhaustive", "scc")
    )
    metafunc.parametrize(
        ("solver", "mode", "depth"), cases, ids=tuple(f"depth{depth}-{mode}-{solver}" for solver, mode, depth in cases)
    )


def test_search_recovery(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, solver: str, mode: str, depth: int
) -> None:
    benchmark.group = f"recovery-{mode}-depth{depth}"
    benchmark.extra_info["timing_scope"] = "cold analyser construction and two successive analysis/realisation requests"
    benchmark_trials(
        benchmark,
        lambda: RecoveryTrial.prepare(solver, mode, depth),
        rounds=request.config.getoption("search_rounds"),
    )
    if request.config.getoption("search_counters"):
        trial = RecoveryTrial.prepare(solver, mode, depth)
        try:
            resources, counts = instrument(solver)
            with resources:
                trial.analyse()
            benchmark.extra_info["instrumented"] = trial.check() | {"counts": dict(counts)}
        finally:
            trial.close()
