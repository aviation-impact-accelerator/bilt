import pytest  # noqa: TC002 - runtime annotation inspection
from benchmark_trials import benchmark_trials
from linear_models import LinearCase, LinearTrial, linear_cases
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime annotation inspection


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    metafunc.parametrize(
        "linear_case", linear_cases(extended=metafunc.config.getoption("search_large")), ids=lambda c: c.label
    )


def test_coupled_linear_evaluation(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, linear_case: LinearCase
) -> None:
    benchmark.group = "linear-evaluation-" + linear_case.label.replace("-hinted", "").replace("-inferred", "")
    benchmark.extra_info["timing_scope"] = (
        "warm public evaluate: cache lookup, alignment, solve, lifting and synchronization"
    )
    benchmark_trials(
        benchmark,
        lambda: LinearTrial.prepare(linear_case, "evaluation"),
        rounds=request.config.getoption("search_rounds"),
    )
