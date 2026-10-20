import os
import typing

import pytest
from structural_search import SearchCase, SearchTrial, count_search

if typing.TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.parametrize("rectangular", (False, True), ids=("square", "rectangular"))
@pytest.mark.parametrize("layout", ("links", "arithmetic", "siblings"))
@pytest.mark.parametrize("coupled", (False, True), ids=("acyclic", "coupled"))
@pytest.mark.parametrize("depth", (0, 1, 2), ids=lambda depth: f"depth{depth}")
@pytest.mark.parametrize("solver", ("exhaustive", "scc"))
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
    active: SearchTrial | None = None
    verified_rounds = 0

    def setup() -> tuple[tuple[SearchTrial], dict[str, object]]:
        nonlocal active
        if active is not None:
            active.close()
        active = SearchTrial.prepare(case)
        return (active,), {}

    def teardown(trial: SearchTrial) -> None:
        nonlocal verified_rounds
        try:
            benchmark.extra_info.update(trial.check())
            verified_rounds += 1
            benchmark.extra_info["verified_rounds"] = verified_rounds
        finally:
            trial.close()

    try:
        benchmark.pedantic(
            SearchTrial.analyse,
            setup=setup,
            teardown=teardown,
            rounds=request.config.getoption("search_rounds"),
        )
        # pytest-benchmark's disabled mode skips its teardown callback.
        if active is not None and not active.checked:
            teardown(active)
    finally:
        # Unexpected exceptions in setup/analysis must not leak contexts or patches.
        if active is not None:
            active.close()
    if request.config.getoption("search_counters"):
        benchmark.extra_info["instrumented"] = count_search(case)
