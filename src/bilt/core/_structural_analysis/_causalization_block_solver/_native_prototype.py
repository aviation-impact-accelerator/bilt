"""Opt-in native planning experiment; Python remains the reference engine."""
# ruff: noqa
import atexit
import json
import os
import time
from collections import Counter

from bilt._native import NativeStructuralModel
from bilt.core._equation import BasicEquation
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock, PlanStep
from bilt.core._structural_analysis._realize import (
    BlockRealizationFailure, RealizationFailure, realize_plan,
)

MODE = os.environ.get("BILT_EXPERIMENTAL_STRUCTURAL_ENGINE", "python")
STATS: Counter[str] = Counter()


def report() -> None:
    if MODE == "rust":
        print("BILT_NATIVE_STATS " + json.dumps(dict(STATS), sort_keys=True))


if MODE == "rust":
    atexit.register(report)


class NativePlanner:
    def __init__(self, incidence) -> None:
        start = time.perf_counter_ns()
        self.names: dict[str, int] = {}
        self.reverse: list[str] = []
        self.operations = []
        contexts = []

        def port(name: str) -> int:
            if name not in self.names:
                self.names[name] = len(self.reverse)
                self.reverse.append(name)
            return self.names[name]

        def visit(graph, prefix: str, ancestors: frozenset[int]) -> int:
            context = len(contexts)
            rows = []
            contexts.append(rows)
            for node in graph.constituent_nodes:
                operation = graph.graph[node]
                index = len(rows)
                if context == 0:
                    self.operations.append(operation)
                names = set(operation.__bilt_parameters__)
                output = None
                child = None
                if isinstance(operation, BasicTransform):
                    names.add(operation.__bilt_name__)
                    output = port(prefix + operation.__bilt_name__)
                elif isinstance(operation, SubSystemConstituent):
                    implementation = operation.implementation
                    if id(implementation) in ancestors:
                        raise ValueError("recursive containment")
                    child = visit(
                        implementation.structural_analysis().incidence,
                        prefix + operation.__bilt_name__ + ".",
                        ancestors | {id(implementation)},
                    )
                elif not isinstance(operation, BasicEquation):
                    raise ValueError("unsupported constituent")
                rows.append((index, [port(prefix + p) for p in sorted(names)], output, child))
            return context

        visit(incidence, "", frozenset())
        self.model = NativeStructuralModel(contexts)
        STATS["model_build_ns"] += time.perf_counter_ns() - start
        STATS["models"] += 1

    def plan(self, available, targets, failures):
        if not all(p in self.names for p in targets):
            return None
        blocked = [
            (index, self.names[step.output])
            for step in failures
            for index, operation in enumerate(self.operations)
            if step.operation is operation and step.output in self.names
        ]
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


def attempt(analyser, request, targets):
    """Return a realised native candidate or None; never persist experimental failures."""
    if MODE != "rust":
        return None
    STATS["requests"] += 1
    if not hasattr(analyser, "_experimental_native"):
        try:
            analyser._experimental_native = NativePlanner(analyser.full_incidence_graph)
        except ValueError:
            analyser._experimental_native = None
    native = analyser._experimental_native
    if native is None:
        STATS["unsupported_model"] += 1
        return None
    available = set(request.input_domains)
    blocks = native.plan(available, targets, analyser._failures)
    if blocks is None:
        STATS["structural_fallback"] += 1
        return None
    start = time.perf_counter_ns()
    result = realize_plan(AnalysisPlan(request=request, blocks=blocks), analyser.backend, block_failures={})
    STATS["realization_ns"] += time.perf_counter_ns() - start
    if isinstance(result, (RealizationFailure, BlockRealizationFailure)):
        STATS["realization_fallback"] += 1
        return None
    if not set(targets) <= ({item.port for item in result.outputs} | available):
        STATS["missing_output_fallback"] += 1
        return None
    STATS["native_success"] += 1
    return result
