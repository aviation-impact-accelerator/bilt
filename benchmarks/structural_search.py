"""Compare cold structural search: uv run python benchmarks/structural_search.py --counters."""

# Standalone CLI: deferred Bilt imports keep controller startup independent.
# ruff: noqa: INP001, PLC0415, T201

import argparse
import functools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import typing
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from bilt import Implementation
    from bilt.core._model_constituent import BiltModelConstituent


class TrialResult(typing.TypedDict, total=False):
    """One completed or timed-out worker result."""

    status: str
    seconds: float
    wall_limit_seconds: float
    error_type: str
    message: str
    value: object
    counts: dict[str, int]


def build_model(*, coupled: bool, depth: int) -> Implementation:
    """Construct a flat or recursively wrapped model with fixed scalar inputs."""
    from bilt import Implementation, Interface, equation, field, system

    @system
    class Ports(metaclass=Interface):
        a: float = field()
        b: float = field()
        x: float = field()
        y: float = field()

    @equation
    def total(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def difference(b: float, x: float, y: float) -> bool:
        return x - y == b

    @equation
    def explicit_x(a: float, b: float, x: float) -> bool:
        return x == a + b

    @equation
    def explicit_y(x: float, y: float) -> bool:
        return y == x + 1

    implementation = Implementation(Ports, [total, difference] if coupled else [explicit_x, explicit_y])
    for _ in range(depth):

        @equation
        def a_link(a: float, inner: Ports) -> bool:
            return inner.a == a

        @equation
        def b_link(b: float, inner: Ports) -> bool:
            return inner.b == b

        @equation
        def x_link(x: float, inner: Ports) -> bool:
            return x == inner.x

        implementation = Implementation(
            Ports,
            [a_link, b_link, x_link],
            internal_implementations={"inner": implementation},
        )
    return implementation


def instrument(solver: str) -> tuple[ExitStack, Counter[str]]:
    """Count search operations without adding overhead to timing-only trials."""
    from bilt.core._structural_analysis._causalization_block_solver import _backend as scc
    from bilt.core._structural_analysis._exhaustive._backend import ExhaustiveStructuralAnalyser
    from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph

    stack = ExitStack()
    counts: Counter[str] = Counter()
    queries: set[tuple[int, frozenset[str], str]] = set()
    factory = ExhaustiveStructuralAnalyser if solver == "exhaustive" else scc.CausalizationStructuralAnalyzer
    original_required = typing.cast(
        "Callable[[object, set[str], str], frozenset[str] | None]", factory.required_inputs_for_target
    )

    def required(self: object, available: set[str], output: str) -> frozenset[str] | None:
        counts["child_route_queries"] += 1
        key = id(self), frozenset(available), output
        if key not in queries:
            queries.add(key)
            counts["distinct_child_route_queries"] += 1
        return original_required(self, available, output)

    def counted(name: str, function: Callable[..., object]) -> Callable[..., object]:
        @functools.wraps(function)
        def call(*args: object, **kwargs: object) -> object:
            counts[name] += 1
            return function(*args, **kwargs)

        return call

    stack.enter_context(patch.object(factory, "required_inputs_for_target", required))
    if solver == "scc":
        for name, label in (
            ("find_matching", "execution_matching_calls"),
            ("settle_assignments", "settlement_calls"),
            ("realize_plan", "realisation_attempts"),
            ("validate_request_structure", "structural_validation_calls"),
        ):
            stack.enter_context(patch.object(scc, name, counted(label, getattr(scc, name))))
        original_plan = typing.cast(
            "Callable[[object, dict[str, BiltModelConstituent], set[str], str], object]",
            scc.CausalizationStructuralAnalyzer._plan_steps,  # noqa: SLF001 - benchmark instrumentation
        )

        def plan(
            self: object, assignments: dict[str, BiltModelConstituent], available: set[str], output: str
        ) -> object:
            counts["candidate_checks"] += 1
            if output not in available and output not in assignments:
                counts["candidates_without_target_assignment"] += 1
            return original_plan(self, assignments, available, output)

        stack.enter_context(patch.object(factory, "_plan_steps", plan))
    else:
        from bilt.core._structural_analysis._exhaustive import _backend as exhaustive

        stack.enter_context(
            patch.object(BipartiteGraph, "search", counted("recursive_search_calls", BipartiteGraph.search))
        )
        stack.enter_context(
            patch.object(exhaustive, "realize_plan", counted("realisation_attempts", exhaustive.realize_plan))
        )
    return stack, counts


def _source_path() -> str:
    import bilt

    return str(Path(bilt.__file__).resolve())


def worker(args: argparse.Namespace) -> None:
    """Run one isolated cold trial; check execution results outside its timer."""
    from bilt._backend._python._backend import PYTHON_BACKEND
    from bilt.core._errors import BiltAnalysisError
    from bilt.core._evaluation_context import EvaluationContext
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._structural_analysis._causalization_block_solver._backend import (
        CausalizationStructuralAnalyzer,
    )
    from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
    from bilt.core._system import _implementation as implementation_module
    from bilt.symbolic._domains import Domain

    factory = ExhaustiveStructuralAnalyser if args.solver == "exhaustive" else CausalizationStructuralAnalyzer
    # Use the same solver in every child, through the existing test seam.
    with patch.object(implementation_module, "ExhaustiveStructuralAnalyser", factory):
        implementation = build_model(coupled=args.case == "coupled", depth=args.depth)
        request = EvaluationRequest(output="x", input_domains={"a": Domain.scalar(), "b": Domain.scalar()})
        stack, counts = instrument(args.solver) if args.count_worker else (ExitStack(), Counter())
        result: TrialResult
        with stack, EvaluationContext():
            start = time.perf_counter()
            try:
                case = implementation.analyser_for(PYTHON_BACKEND).analyse(request)
            except BiltAnalysisError as error:
                result = {
                    "status": "rejected",
                    "seconds": time.perf_counter() - start,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            else:
                elapsed = time.perf_counter() - start
                actual = PYTHON_BACKEND.chain_case(case).callable(a=4.0, b=2.0)
                expected = 3.0 if args.case == "coupled" else 6.0
                if actual != expected:
                    message = f"Expected {expected}, received {actual}."
                    raise AssertionError(message)
                result = {"status": "success", "seconds": elapsed, "value": actual}
        expected_status = "rejected" if args.case == "coupled" else "success"
        if result["status"] != expected_status:
            message = f"Unexpected benchmark outcome: {result}"
            raise AssertionError(message)
        if args.count_worker:
            result["counts"] = dict(counts)
        print(json.dumps(result), flush=True)


def trial(args: argparse.Namespace, solver: str, case: str, depth: int, *, count: bool = False) -> TrialResult:
    """Bound total worker wall time, while the worker times analysis separately."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--solver",
        solver,
        "--case",
        case,
        "--depth",
        str(depth),
    ]
    if count:
        command.append("--count-worker")
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.hash_seed)
    try:
        result = subprocess.run(  # noqa: S603 - invokes this script with validated CLI choices
            command, capture_output=True, text=True, timeout=args.timeout, check=True, env=env
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "wall_limit_seconds": args.timeout}
    except subprocess.CalledProcessError as error:
        message = f"Worker failed:\n{error.stdout}\n{error.stderr}"
        raise RuntimeError(message) from error
    return typing.cast("TrialResult", json.loads(result.stdout))


def main() -> None:
    """Run timing trials and optional separate operation-count trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--output", default="structural-search-results.json")
    parser.add_argument("--counters", action="store_true", help="Run a separate instrumented trial per case.")
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--solver", choices=("all", "exhaustive", "scc"), default="all")
    parser.add_argument("--case", choices=("all", "acyclic", "coupled"), default="all")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--count-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.repeats < 1 or args.timeout <= 0 or args.max_depth < 0:
        parser.error("repeats and timeout must be positive; max-depth must be nonnegative")
    rows = []
    cases = ("acyclic", "coupled") if args.case == "all" else (args.case,)
    solvers = ("exhaustive", "scc") if args.solver == "all" else (args.solver,)
    for case in cases:
        for depth in range(args.max_depth + 1):
            for solver in solvers:
                trials = []
                for _ in range(args.repeats):
                    result = trial(args, solver, case, depth)
                    trials.append(result)
                    if result["status"] == "timeout":
                        break
                seconds = [r["seconds"] for r in trials if "seconds" in r]
                median = statistics.median(seconds) if seconds else None
                row: dict[str, object] = {
                    "case": case,
                    "depth": depth,
                    "solver": solver,
                    "trials": trials,
                    "median_seconds": median,
                    "completed_trials": len(seconds),
                }
                if args.counters:
                    row["instrumented"] = trial(args, solver, case, depth, count=True)
                rows.append(row)
                timing = f"{median * 1000:.3f} ms" if median is not None else f"wall timeout ({args.timeout:g} s)"
                print(
                    f"{case:8} depth={depth} {solver:10} {trials[-1]['status']:8} {timing}",
                    flush=True,
                )
    report = {
        "hash_seed": args.hash_seed,
        "timing_scope": "cold analyser construction, analysis and realisation; excludes imports, model build and execution",
        "source_path": _source_path(),
        "python": sys.version,
        "platform": platform.platform(),
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
