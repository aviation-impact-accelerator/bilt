import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("structural search")
    group.addoption("--search-rounds", type=int, default=5, help="Fresh single-iteration rounds per benchmark.")
    group.addoption("--search-counters", action="store_true", help="Run separate untimed search counters.")
    group.addoption(
        "--search-extended",
        "--search-large",
        dest="search_large",
        action="store_true",
        help="Run the full structural benchmark matrix; --search-large is a compatibility alias.",
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("search_rounds") < 1:
        msg = "--search-rounds must be positive"
        raise pytest.UsageError(msg)
