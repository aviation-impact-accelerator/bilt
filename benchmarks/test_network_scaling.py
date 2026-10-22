import pytest  # noqa: TC002
from benchmark_trials import benchmark_trials
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002 - runtime annotation inspection
from scale_models import ScaleCase, ScaleTrial
from structural_search import instrument


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    sizes = {"chain": (4,), "tree": (1,), "parallel": (1,), "series": (1,)}
    if metafunc.config.getoption("search_large"):
        sizes = {"chain": (4, 8, 16), "tree": (1, 2, 3), "parallel": (0, 1, 2), "series": (1, 2)}
    cases = [
        ScaleCase(solver, topology, depth)
        for topology, depths in sizes.items()
        for depth in depths
        for solver in ("exhaustive", "scc")
    ]
    if metafunc.config.getoption("search_large"):
        # Preserve the existing SCC-only stress endpoints without adding new workloads.
        cases.extend(
            ScaleCase("scc", topology, depth) for topology, depth in (("chain", 32), ("tree", 4), ("parallel", 3))
        )
    metafunc.parametrize("scale_case", cases, ids=lambda case: f"{case.solver}-{case.topology}-depth{case.depth}")


def test_search_scaling(benchmark: BenchmarkFixture, request: pytest.FixtureRequest, scale_case: ScaleCase) -> None:
    case = scale_case
    benchmark.group = f"scaling-{case.topology}-depth{case.depth}"
    benchmark.extra_info["timing_scope"] = "cold analyser construction, analysis and realisation"
    benchmark_trials(
        benchmark,
        lambda: ScaleTrial.prepare_scale(case),
        rounds=request.config.getoption("search_rounds"),
    )
    if request.config.getoption("search_counters"):
        trial = ScaleTrial.prepare_scale(case)
        try:
            resources, counts = instrument(case.solver)
            with resources:
                trial.analyse()
            benchmark.extra_info["instrumented"] = trial.check() | {"counts": dict(counts)}
        finally:
            trial.close()
