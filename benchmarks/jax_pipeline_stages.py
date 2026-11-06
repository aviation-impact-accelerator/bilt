"""Profile JAX tracing, lowering, compilation and execution for Bilt models.

Run explicitly; these are informational timings, never test pass/fail thresholds.
"""

# ruff: noqa: INP001, T201 - standalone benchmark CLI reports its timings to stdout

import argparse
import json
import platform
import statistics
import time
import typing
from pathlib import Path

import jax
from linear_models import LinearCase, LinearTrial

from bilt.core._port import mangle_inputs
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan

if typing.TYPE_CHECKING:
    from collections.abc import Callable


CASES = {
    "electrical-shared": LinearCase(family="electrical", shared=True, hinted=False, backend_name="jax"),
    "electrical-pointwise": LinearCase(family="electrical", shared=False, hinted=False, backend_name="jax"),
    "thermal-shared": LinearCase(family="thermal", shared=True, hinted=False, backend_name="jax"),
    "thermal-pointwise": LinearCase(family="thermal", shared=False, hinted=False, backend_name="jax"),
}


def _timed[T](action: Callable[[], T]) -> tuple[T, float]:
    """Return one result and elapsed wall-clock milliseconds."""
    start = time.perf_counter()
    result = action()
    return result, (time.perf_counter() - start) * 1000


def _synchronise(value: object) -> None:
    """Wait for all returned arrays and ordered JAX debug callbacks."""
    jax.block_until_ready(value)
    jax.effects_barrier()


def profile_case(case: LinearCase, *, warm_runs: int) -> dict[str, object]:
    """Profile a fresh real Bilt model using JAX's explicit compilation stages."""
    trial = LinearTrial.prepare(case, "analysis")
    try:
        backend_callable, bilt_ms = _timed(lambda: trial.implementation.get_evaluation(trial.request, case.backend))

        # Input validation, units, alignment, transfer and dispatch are not
        # part of the trace/lower/compile/execute timings.
        aligned = case.backend.apply_alignment(lower_inputs(trial.inputs), generate_alignment_plan(trial.inputs))
        mangled = mangle_inputs(aligned)
        arguments = {name: jax.device_put(mangled[name]) for name in backend_callable.parameters}
        jax.block_until_ready(arguments)

        # Bilt's JAX backend creates an outer jax.jit wrapper when chaining.
        # Trace once to a jaxpr, then lower and compile the SAME staged graph.
        wrapped = typing.cast("typing.Any", backend_callable.callable)
        traced, trace_ms = _timed(lambda: wrapped.trace(**arguments))
        lowered, lower_ms = _timed(traced.lower)
        executable, compile_ms = _timed(lowered.compile)
        _, first_execution_ms = _timed(lambda: _synchronise(executable(**arguments)))

        warmed_execution_ms: list[float] = []
        for _ in range(warm_runs):
            _, elapsed = _timed(lambda: _synchronise(executable(**arguments)))
            warmed_execution_ms.append(elapsed)

        # Check through the public API, outside all measured stages, using
        # analytic physical reference values rather than another solver.
        for output, expected in trial.expected.items():
            result = trial.implementation.evaluate(trial.inputs, output, case.backend)
            _synchronise(result.values if hasattr(result, "values") else getattr(result, "value", result))
            expected.check(result)

        return {
            "case": case.label,
            "target": trial.request.output,
            "coordinates": case.coordinates,
            "bilt_planning_ms": round(bilt_ms, 3),
            "jax_trace_ms": round(trace_ms, 3),
            "jax_lower_ms": round(lower_ms, 3),
            "xla_compile_ms": round(compile_ms, 3),
            "first_execution_ms": round(first_execution_ms, 3),
            "warm_execution_median_ms": round(statistics.median(warmed_execution_ms), 3),
            "warm_execution_samples_ms": [round(value, 3) for value in warmed_execution_ms],
        }
    finally:
        trial.close()


def main() -> None:
    """Run the opt-in profile and optionally save machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--json", type=Path, help="Optional output path for the JSON report.")
    args = parser.parse_args()
    if args.warm_runs < 1:
        parser.error("--warm-runs must be positive")

    selected = CASES.values() if args.case == "all" else (CASES[args.case],)
    measurements = [profile_case(case, warm_runs=args.warm_runs) for case in selected]
    report = {
        "environment": {
            "jax_version": jax.__version__,
            "device": str(jax.devices()[0]),
            "python": platform.python_version(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "warm_runs": args.warm_runs,
        },
        "cases": measurements,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.json is not None:
        args.json.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
