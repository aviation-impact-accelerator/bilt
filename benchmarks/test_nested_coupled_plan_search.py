"""Bounded search benchmark comparing equivalent equations in a nested system."""

import pytest
from benchmark_trials import benchmark_trials
from nested_coupled_plan_search import NestedCoupledTrial
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime annotation inspection


@pytest.mark.parametrize("coupled", (False, True), ids=("explicit", "coupled"))
def test_nested_coupled_plan_search(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, *, coupled: bool
) -> None:
    """Compare planning for equivalent coupled and explicit formulations."""
    benchmark.group = "nested-coupled-scc"
    benchmark.extra_info["timing_scope"] = "cold structural analysis until success or bounded plan-search budget"
    benchmark.extra_info["limit"] = "1000 plan construction attempts, including nested child requests"
    benchmark_trials(
        benchmark,
        lambda: NestedCoupledTrial.prepare(coupled=coupled),
        rounds=request.config.getoption("search_rounds"),
    )
