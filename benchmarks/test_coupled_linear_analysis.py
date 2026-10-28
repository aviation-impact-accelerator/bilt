import pytest  # noqa: TC002 - runtime annotation inspection
from benchmark_trials import benchmark_trials
from linear_models import LinearCase, LinearTrial, linear_cases
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime annotation inspection
from structural_search import instrument


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    metafunc.parametrize(
        "linear_case", linear_cases(extended=metafunc.config.getoption("search_large")), ids=lambda c: c.label
    )


def test_coupled_linear_analysis(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, linear_case: LinearCase
) -> None:
    benchmark.group = "linear-analysis-" + linear_case.label.replace("-hinted", "").replace("-inferred", "")
    benchmark.extra_info["timing_scope"] = "cold Bilt analyser construction, planning, realisation and chaining"
    benchmark_trials(
        benchmark,
        lambda: LinearTrial.prepare(linear_case, "analysis"),
        rounds=request.config.getoption("search_rounds"),
    )
    if request.config.getoption("search_counters"):
        trial = LinearTrial.prepare(linear_case, "analysis")
        try:
            resources, counts = instrument("scc")
            with resources:
                trial.analyse()
            benchmark.extra_info["instrumented"] = trial.check() | {"counts": dict(counts)}
        finally:
            trial.close()
