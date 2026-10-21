import os
import typing
import pytest

from benchmark_trials import benchmark_trials
from structural_search import SearchCase, SearchTrial, count_search

if typing.TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    extended = metafunc.config.getoption("search_large")
    cases = [
        SearchCase(solver, layout, coupled=coupled, depth=depth, rectangular=rectangular)
        for layout in ("links", "arithmetic", "siblings")
        for rectangular in (False, True)
        for coupled in (False, True)
        for depth in ((0, 1, 2) if extended else (1,))
        for solver in ("exhaustive", "scc")
    ]
    if not extended:
        # Retain a primitive baseline and the original deep rectangular regression.
        cases.extend(
            SearchCase(solver, "links", coupled=coupled, depth=0, rectangular=False)
            for coupled in (False, True)
            for solver in ("exhaustive", "scc")
        )
        cases.extend(
            SearchCase(solver, "siblings", coupled=True, depth=2, rectangular=True) for solver in ("exhaustive", "scc")
        )
    metafunc.parametrize(
        "search_case",
        cases,
        ids=lambda case: (
            f"{case.solver}-depth{case.depth}-{'coupled' if case.coupled else 'acyclic'}-"
            f"{case.layout}-{'rectangular' if case.rectangular else 'square'}"
        ),
    )


def test_structural_search(
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    search_case: SearchCase,
) -> None:
    case = search_case
    layout, depth = case.layout, case.depth
    coupled, rectangular = case.coupled, case.rectangular
    benchmark.group = (
        f"{'rectangular' if rectangular else 'square'}-{layout}-{'coupled' if coupled else 'acyclic'}-depth{depth}"
    )
    benchmark.extra_info.update(
        {
            "timing_scope": "cold analyser construction, analysis and realisation",
            "isolation": "fresh model and evaluation context per round; shared interpreter",
            "hash_seed": os.environ.get("PYTHONHASHSEED", "random"),
        }
    )
    benchmark_trials(
        benchmark,
        lambda: SearchTrial.prepare(case),
        rounds=request.config.getoption("search_rounds"),
    )
    if request.config.getoption("search_counters"):
        benchmark.extra_info["instrumented"] = count_search(case)
