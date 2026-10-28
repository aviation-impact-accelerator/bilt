"""Compare single and sequential attribute access, including retained tensor storage.

Run on a revision before this feature as well as the current one to measure the
execution/memory tradeoff. Compilation is warmed before timing; each timed call
uses a fresh instance. Retained bytes describe cached tensors, not peak memory.
"""

import typing

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002

from bilt import Implementation, Interface, Tensor, dim, equation, field, system, tensor, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._system._instance import SystemInstance

sample = dim("cache_benchmark_sample")


@system
class Model(metaclass=Interface):
    values: Tensor = field()
    shared: Tensor = field()
    total: float = field()
    average: float = field()


@transform
def shared(values: Tensor) -> Tensor:
    return values * values


@equation
def total_eq(shared: Tensor, total: float) -> bool:
    return total == shared.sum(sample)


@equation
def average_eq(shared: Tensor, average: float) -> bool:
    return average == shared.mean(sample)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize("sequential", (False, True), ids=("single", "sequential"))
def test_attribute_access(benchmark: BenchmarkFixture, backend: Backend, *, sequential: bool) -> None:
    implementation = Implementation(Model, [shared, total_eq, average_eq])
    values = tensor(np.ones(1_000_000), over=sample)

    def evaluate() -> Model:
        instance = Model(values=values)
        # Conversion synchronizes scalar JAX results before stopping the timer.
        assert float(instance.total) == pytest.approx(1_000_000.0)
        if sequential:
            assert float(instance.average) == pytest.approx(1.0)
        return instance

    with backend, implementation:
        evaluate()
        instance = benchmark.pedantic(evaluate, rounds=20, iterations=1)

    cached = typing.cast("SystemInstance", instance).__bilt_system_instance_variables__
    benchmark.extra_info["retained_intermediate_bytes"] = sum(
        value.values.nbytes for name, value in cached.items() if name != "values" and isinstance(value, Tensor)
    )
