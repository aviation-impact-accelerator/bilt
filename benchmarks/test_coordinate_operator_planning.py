"""Benchmark inverse source-to-target planning over increasingly large tensors."""

import pytest
from benchmark_trials import benchmark_trials
from coordinate_operator_planning import CoordinateOperatorTrial
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime fixture reflection


@pytest.mark.parametrize("size", (2, 4, 8, 12))
def test_coordinate_operator_planning(
    benchmark: BenchmarkFixture, request: pytest.FixtureRequest, size: int
) -> None:
    """Time fresh structural analysis and realisation, checking every solve."""
    benchmark.group = "coordinate-operator-planning"
    benchmark.extra_info["timing_scope"] = "cold structural analysis and realisation"
    benchmark_trials(
        benchmark,
        lambda: CoordinateOperatorTrial.prepare(size),
        rounds=request.config.getoption("search_rounds"),
    )
