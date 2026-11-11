"""End-to-end coupled linear planning for three and four unknowns."""

import pytest
from benchmark_trials import benchmark_trials
from coupled_group_planning import CoupledGroupTrial
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime annotation inspection


@pytest.mark.parametrize("size", (3, 4))
def test_coupled_group_planning(benchmark: BenchmarkFixture, request: pytest.FixtureRequest, size: int) -> None:
    """Time fresh linear planning and verify the computed x after each round."""
    benchmark.group = "coupled-groups"
    benchmark.extra_info["timing_scope"] = "cold structural analysis and realisation"
    benchmark_trials(
        benchmark, lambda: CoupledGroupTrial.prepare(size), rounds=request.config.getoption("search_rounds")
    )
