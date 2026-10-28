"""Time engineering report sessions, separating cold compilation from warm execution."""

import dataclasses
import time
import typing
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest
from intermediate_workloads import Workload, prepare_workload
from pytest_benchmark.fixture import BenchmarkFixture  # noqa: TC002

from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._system import _implementation as implementation_module
from bilt.core._system._sentinel import MISSING
from bilt.core._tensors._tensors import Tensor

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._system._instance import SystemInstance


def read(instance: object, name: str) -> object:
    value = getattr(instance, name)
    if callable(synchronize := getattr(value, "block_until_ready", None)):
        synchronize()
    return value


def storage_bytes(values: list[np.ndarray], excluding: list[np.ndarray]) -> int:
    seen = excluding.copy()
    total = 0
    for value in values:
        root = value
        while isinstance(root.base, np.ndarray):
            root = root.base
        if not any(np.shares_memory(root, other) for other in seen):
            total += root.nbytes
            seen.append(root)
    return total


@dataclasses.dataclass
class Session:
    workload: Workload
    instance: object
    results: dict[str, object] = dataclasses.field(default_factory=dict)
    first_ns: int = 0
    later_ns: int = 0
    checked: bool = False

    def execute(self) -> None:
        first, *later = self.workload.expected
        start = time.perf_counter_ns()
        self.results[first] = read(self.instance, first)
        middle = time.perf_counter_ns()
        for name in later:
            self.results[name] = read(self.instance, name)
        end = time.perf_counter_ns()
        self.first_ns = middle - start
        self.later_ns = end - middle

    def check(self) -> dict[str, int]:
        for name, expected in self.workload.expected.items():
            result = self.results[name]
            values = result.values if isinstance(result, Tensor) else result
            np.testing.assert_allclose(np.asarray(values), np.asarray(expected), rtol=2e-5, atol=1e-7)
        variables = typing.cast("SystemInstance", self.instance).__bilt_system_instance_variables__
        tensors = {
            name: value.values
            for name, value in variables.items()
            if name not in self.workload.inputs and isinstance(value, Tensor)
        }
        inputs = [value.values for value in self.workload.inputs.values() if isinstance(value, Tensor)]
        requested = [value for name, value in tensors.items() if name in self.workload.expected]
        # Exclude buffers already supplied by the caller, and count aliasing port
        # views only once. This measures retained tensor storage, not peak RSS.
        self.checked = True
        return {
            "retained_tensor_bytes": storage_bytes(list(tensors.values()), inputs),
            "retained_intermediate_bytes": storage_bytes(list(tensors.values()), inputs + requested),
            "cached_ports": sum(value is not MISSING for value in variables.values()),
        }


@pytest.mark.parametrize("solver", ("exhaustive", "causalization"))
@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize("size", (1, 10_000), ids=("scalar", "batch10000"))
@pytest.mark.parametrize("family", ("geometry", "series", "mixed"))
@pytest.mark.parametrize("temperature", ("cold", "warm"))
def test_reporting_session(  # noqa: PLR0913, PLR0917
    benchmark: BenchmarkFixture,
    request: pytest.FixtureRequest,
    solver: str,
    backend: Backend,
    size: int,
    family: str,
    temperature: str,
) -> None:
    factory = ExhaustiveStructuralAnalyser if solver == "exhaustive" else CausalizationStructuralAnalyzer
    warm_workload: Workload | None = None
    active: Session | None = None
    samples: list[tuple[int, int]] = []
    with ExitStack() as resources:
        resources.enter_context(patch.object(implementation_module, "ExhaustiveStructuralAnalyser", factory))
        resources.enter_context(typing.cast("typing.ContextManager[object]", backend))
        if temperature == "warm":
            warm_workload = prepare_workload(family, size)
            with warm_workload.implementation:
                warmup = Session(warm_workload, warm_workload.implementation.system(**warm_workload.inputs))
                warmup.execute()
                warmup.check()

        def setup() -> tuple[tuple[Session], dict[str, object]]:
            nonlocal active
            workload = warm_workload or prepare_workload(family, size)
            session = Session(workload, workload.implementation.system(**workload.inputs))
            active = session
            return (session,), {}

        def execute(session: Session) -> None:
            with session.workload.implementation:
                session.execute()

        def teardown(session: Session) -> None:
            benchmark.extra_info.update(session.check())
            samples.append((session.first_ns, session.later_ns))

        rounds = request.config.getoption("execution_cold_rounds" if temperature == "cold" else "execution_warm_rounds")
        benchmark.pedantic(execute, setup=setup, teardown=teardown, rounds=rounds, iterations=1)
        # Disabled benchmarks omit teardown; still check every returned value.
        if active is not None and not active.checked:
            teardown(active)
        benchmark.extra_info.update(
            first_output_ns=[first for first, _ in samples],
            subsequent_outputs_ns=[later for _, later in samples],
            outputs=list(active.workload.expected) if active is not None else [],
            family=family,
            size=size,
            solver=solver,
            backend="jax" if backend is JAX_BACKEND else "python",
            temperature=temperature,
        )
