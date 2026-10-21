import typing
import pytest

from benchmark_trials import benchmark_trials
from difficult_models import DifficultCase, DifficultTrial
from structural_search import instrument

if typing.TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    sizes = {"reverse": (1,), "heat": (1,), "flow": (1,), "mixed": (2,), "unanchored": (1,)}
    if metafunc.config.getoption("search_large"):
        sizes = {
            "reverse": (1, 2, 3),
            "heat": (1, 2, 3),
            "flow": (1, 2, 3),
            "mixed": (2, 3, 4),
            "unanchored": (0, 1, 2),
        }
    cases = [
        DifficultCase(solver, scenario, depth)
        for scenario, depths in sizes.items()
        for depth in depths
        for solver in ("exhaustive", "scc")
    ]
    metafunc.parametrize("difficult_case", cases, ids=lambda case: f"{case.solver}-{case.scenario}-depth{case.depth}")


def test_difficult_search(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, difficult_case: DifficultCase
) -> None:
    case = difficult_case
    benchmark.group = f"difficult-{case.scenario}-depth{case.depth}"
    benchmark.extra_info["timing_scope"] = "cold analyser construction, analysis and realisation"
    benchmark_trials(
        benchmark,
        lambda: DifficultTrial.prepare_difficult(case),
        rounds=request.config.getoption("search_rounds"),
    )
    if request.config.getoption("search_counters"):
        trial = DifficultTrial.prepare_difficult(case)
        try:
            resources, counts = instrument(case.solver)
            with resources:
                trial.analyse()
            benchmark.extra_info["instrumented"] = trial.check() | {"counts": dict(counts)}
        finally:
            trial.close()
