"""Run fresh, single-analysis trials and check every result outside timing."""

# ruff: noqa: INP001 - benchmark helper, not an installed package

import os
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_benchmark.fixture import BenchmarkFixture


class Trial(typing.Protocol):
    """The lifecycle required by the benchmark runner."""

    checked: bool

    def analyse(self) -> None:
        """Perform the timed work once."""
        ...

    def check(self) -> dict[str, object]:
        """Verify the result and return report metadata."""
        ...

    def close(self) -> None:
        """Release contexts and restore patches."""
        ...


def benchmark_trials(benchmark: BenchmarkFixture, prepare: Callable[[], Trial], *, rounds: int) -> None:
    """Ensure disabled timing and unexpected exceptions still check or clean up."""
    benchmark.extra_info.setdefault("isolation", "fresh model and evaluation context per round; shared interpreter")
    benchmark.extra_info.setdefault("hash_seed", os.environ.get("PYTHONHASHSEED", "random"))
    active: Trial | None = None
    verified_rounds = 0

    def setup() -> tuple[tuple[Trial], dict[str, object]]:
        nonlocal active
        if active is not None:
            active.close()
        active = prepare()
        return (active,), {}

    def teardown(trial: Trial) -> None:
        nonlocal verified_rounds
        try:
            benchmark.extra_info.update(trial.check())
            verified_rounds += 1
            benchmark.extra_info["verified_rounds"] = verified_rounds
        finally:
            trial.close()

    try:
        benchmark.pedantic(lambda trial: trial.analyse(), setup=setup, teardown=teardown, rounds=rounds)
        # pytest-benchmark's disabled mode does not invoke teardown.
        if active is not None and not active.checked:
            teardown(active)
    finally:
        if active is not None:
            active.close()
