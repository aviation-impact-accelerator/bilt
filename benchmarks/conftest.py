import pytest
from structural_search import SearchCase


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("structural search")
    group.addoption("--search-rounds", type=int, default=5, help="Fresh single-iteration rounds per benchmark.")
    group.addoption("--search-counters", action="store_true", help="Run separate untimed search counters.")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("search_rounds") < 1:
        msg = "--search-rounds must be positive"
        raise pytest.UsageError(msg)


@pytest.fixture
def search_case(solver: str, layout: str, depth: int, *, coupled: bool, rectangular: bool) -> SearchCase:
    return SearchCase(solver, layout, coupled, depth, rectangular)
