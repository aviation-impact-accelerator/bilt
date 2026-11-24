"""Opt-in native planning experiment; Python remains the reference engine."""

import atexit
import json
import os
import time
import typing
from collections import Counter

from bilt._native import NativeStructuralModel
from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock, PlanStep
from bilt.core._structural_analysis._realize import (
    BlockRealizationFailure,
    RealizationFailure,
    realize_plan,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Set as AbstractSet

    from bilt._backend import BackendCase
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph

type RawRow = tuple[int, list[int], int | None, int | None]

MODE = os.environ.get("BILT_EXPERIMENTAL_STRUCTURAL_ENGINE", "python")
if MODE not in {"python", "rust", "hybrid"}:
    msg = f"Unknown experimental structural engine: {MODE!r}."
    raise ValueError(msg)
STATS: Counter[str] = Counter()
TRACE_BOUNDARIES = "BILT_EXPERIMENTAL_TRACE_BOUNDARIES" in os.environ


def report() -> None:
    if MODE != "python":
        print("BILT_NATIVE_STATS " + json.dumps({"engine": MODE, **STATS}, sort_keys=True))  # noqa: T201


if MODE != "python":
    atexit.register(report)


class NativePlanner:
    def __init__(self, incidence: FullIncidenceGraph) -> None:
        start = time.perf_counter_ns()
        self.names: dict[str, int] = {}
        self.reverse: list[str] = []
        self.operations: list[BiltModelConstituent] = []
        contexts: list[list[RawRow]] = []
        self._visit(incidence, "", frozenset(), contexts)
        self.model = NativeStructuralModel(contexts)
        STATS["model_build_ns"] += time.perf_counter_ns() - start
        STATS["models"] += 1

    def _port(self, name: str) -> int:
        if name not in self.names:
            self.names[name] = len(self.reverse)
            self.reverse.append(name)
        return self.names[name]

    def _visit(
        self,
        graph: FullIncidenceGraph,
        prefix: str,
        ancestors: frozenset[int],
        contexts: list[list[RawRow]],
    ) -> int:
        context = len(contexts)
        rows: list[RawRow] = []
        contexts.append(rows)
        for node in graph.constituent_nodes:
            operation = typing.cast("BiltModelConstituent", graph.graph[node])
            index = len(rows)
            if context == 0:
                self.operations.append(operation)
            names = set(operation.__bilt_parameters__)
            output = None
            child = None
            if isinstance(operation, BasicTransform):
                names.add(operation.__bilt_name__)
                output = self._port(prefix + operation.__bilt_name__)
            elif isinstance(operation, SubSystemConstituent):
                implementation = operation.implementation
                if id(implementation) in ancestors:
                    msg = "recursive containment"
                    raise ValueError(msg)
                child = self._visit(
                    implementation.structural_analysis().incidence,
                    prefix + operation.__bilt_name__ + ".",
                    ancestors | {id(implementation)},
                    contexts,
                )
            elif not isinstance(operation, BasicEquation):
                msg = "unsupported constituent"
                raise TypeError(msg)
            rows.append((index, [self._port(prefix + p) for p in sorted(names)], output, child))
        return context

    def plan(
        self,
        available: AbstractSet[str],
        targets: tuple[str, ...],
        failures: Mapping[PlanStep, RealizationFailure],
    ) -> tuple[PlanBlock, ...] | None:
        if not all(p in self.names for p in targets):
            return None
        blocked = [
            (index, self.names[step.output])
            for step in failures
            for index, operation in enumerate(self.operations)
            if step.operation is operation and step.output in self.names
        ]
        if TRACE_BOUNDARIES:
            print(  # noqa: T201
                "NATIVE_REQUEST " + json.dumps({"targets": targets, "available": sorted(available), "ports": self.reverse}),
                flush=True,
            )
        start = time.perf_counter_ns()
        schedule = self.model.plan(
            [self.names[p] for p in available if p in self.names],
            [self.names[p] for p in targets],
            blocked,
        )
        STATS["native_plan_ns"] += time.perf_counter_ns() - start
        if schedule is None:
            return None
        blocks = []
        for index, output, inputs in schedule:
            operation = self.operations[index]
            child_inputs = (
                frozenset(operation.child_port(self.reverse[p]) for p in inputs)
                if isinstance(operation, SubSystemConstituent)
                else None
            )
            blocks.append(PlanBlock((PlanStep(operation, self.reverse[output], child_inputs),)))
        return tuple(blocks)


def _native_planner(analyser: CausalizationStructuralAnalyzer) -> NativePlanner | None:
    if not analyser._experimental_native_initialized:  # noqa: SLF001
        analyser._experimental_native_initialized = True  # noqa: SLF001
        try:
            analyser._experimental_native = NativePlanner(analyser.full_incidence_graph)  # noqa: SLF001
        except TypeError, ValueError:
            analyser._experimental_native = None  # noqa: SLF001
    return analyser._experimental_native  # noqa: SLF001


class NativeStructuralUnsupportedError(RuntimeError):
    """Strict mode cannot yet handle this request; never silently use Python."""


def _unsupported(reason: str, targets: tuple[str, ...]) -> None:
    STATS[reason] += 1
    if MODE == "rust":
        msg = f"Native structural engine: {reason} for {targets!r}."
        raise NativeStructuralUnsupportedError(msg)


def attempt(
    analyser: CausalizationStructuralAnalyzer,
    request: EvaluationRequest,
    targets: tuple[str, ...],
) -> BackendCase | None:
    """Return a realised native candidate or None; never persist experimental failures."""
    if MODE == "python":
        return None
    STATS["requests"] += 1
    native = _native_planner(analyser)
    if native is None:
        _unsupported("unsupported_model", targets)
        return None
    available = set(request.input_domains)
    blocks = native.plan(available, targets, analyser._failures)  # noqa: SLF001
    if blocks is None:
        _unsupported("structural_fallback", targets)
        return None
    start = time.perf_counter_ns()
    result = realize_plan(AnalysisPlan(request=request, blocks=blocks), analyser.backend, block_failures={})
    STATS["realization_ns"] += time.perf_counter_ns() - start
    if isinstance(result, (RealizationFailure, BlockRealizationFailure)):
        _unsupported("realization_fallback", targets)
        return None
    if not set(targets) <= ({item.port for item in result.outputs} | available):
        _unsupported("missing_output_fallback", targets)
        return None
    STATS["native_success"] += 1
    return result


def attempt_required_inputs(
    analyser: CausalizationStructuralAnalyzer, available: set[str], output: str
) -> frozenset[str] | None:
    """Use the native route planner for explicit child-route queries as well."""
    if MODE == "python":
        return None
    STATS["route_requests"] += 1
    if output in available:
        STATS["route_success"] += 1
        return frozenset({output})
    native = _native_planner(analyser)
    blocks = None if native is None else native.plan(available, (output,), analyser._failures)  # noqa: SLF001
    if blocks is None:
        _unsupported("route_fallback", (output,))
        return None
    produced = {step.output for block in blocks for step in block.steps}
    STATS["route_success"] += 1
    return frozenset(
        attribute for block in blocks for step in block.steps for attribute in step.consumed if attribute not in produced
    )
