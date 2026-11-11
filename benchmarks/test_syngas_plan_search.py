"""Bounded syngas search benchmarks, based on the fuels nested-model regression."""

import pytest
from benchmark_trials import benchmark_trials
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime annotation inspection
from syngas_plan_search import SyngasTrial


@pytest.mark.parametrize("coupled", (False, True), ids=("explicit", "coupled"))
def test_syngas_plan_search(benchmark: BenchmarkFixture, request: pytest.FixtureRequest, *, coupled: bool) -> None:
    """Benchmark planning until a correct answer or the candidate-work limit."""
    benchmark.group = "syngas-nested-scc"
    benchmark.extra_info["timing_scope"] = "cold structural analysis until success or bounded plan-search budget"
    benchmark.extra_info["limit"] = "1000 plan construction attempts, including nested child requests"
    benchmark_trials(
        benchmark,
        lambda: SyngasTrial.prepare(coupled=coupled),
        rounds=request.config.getoption("search_rounds"),
    )
