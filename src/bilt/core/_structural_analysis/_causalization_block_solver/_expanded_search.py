"""Search primitive directions and block formulations within the expanded scope."""

import typing

import rustworkx as rx

from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver import _expanded_planning as planning
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._qualified_block import QualifiedPlan, QualifiedStep
from bilt.core._structural_analysis._realize_qualified import QualifiedRealizationFailure, realize_qualified_plan

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCase
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import ExpansionSnapshot
    from bilt.core._structural_analysis._realize_qualified import QualifiedFailureKey


def find_matching(graph: rx.PyGraph) -> set[tuple[int, int]]:
    """Find a maximum primitive assignment, independently instrumentable from structural proofs."""
    return rx.max_weight_matching(graph, max_cardinality=True)


def candidate_assignments(
    graph: rx.PyGraph, failures: set[QualifiedStep], output: str
) -> collections.abc.Iterator[dict[str, QualifiedStep]]:
    """Branch selected matching edges; exclude only failed primitive directions."""
    target = next((node for node in graph.node_indices() if graph[node] == output), None)
    if target is None:
        return
    pending = [graph]
    visited = set()
    while pending:
        candidate = pending.pop()
        for first, second in list(candidate.edge_list()):
            if planning.matching_step(candidate, first, second) in failures:
                candidate.remove_edge(first, second)
        if candidate.degree(target) == 0:
            continue
        edges = list(candidate.edge_list())
        signature = frozenset((min(a, b), max(a, b)) for a, b in edges)
        if signature in visited:
            continue
        visited.add(signature)
        matching = find_matching(candidate)
        assignments = {step.output: step for a, b in matching for step in (planning.matching_step(candidate, a, b),)}
        yield assignments
        if any(planning.matching_step(candidate, a, b) in failures for a, b in edges):
            pending.append(candidate)
            continue
        for first, second in edges:
            step = planning.matching_step(candidate, first, second)
            if assignments.get(step.output) == step:
                alternative = candidate.copy()
                alternative.remove_edge(first, second)
                pending.append(alternative)


def analyse_expanded(request: EvaluationRequest, structure: ExpansionSnapshot, backend: Backend) -> BackendCase:
    """Retry qualified formulations without altering structural validation or child caches."""
    failures: set[QualifiedStep] = set()
    block_failures: dict[QualifiedFailureKey, QualifiedRealizationFailure] = {}
    last_error = None
    for assignments in candidate_assignments(planning.matching_graph(structure), failures, request.output):
        try:
            blocks = planning.plan_blocks(assignments, set(request.input_domains), request.output)
        except BiltAnalysisError as error:
            if last_error is None:
                last_error = error
            continue
        result = realize_qualified_plan(QualifiedPlan(request, blocks), backend, block_failures)
        if not isinstance(result, QualifiedRealizationFailure):
            return result
        last_error = result.cause
        if isinstance(result.failed, QualifiedStep):
            failures.add(result.failed)
    message = f"Cannot compute {request.output} from available inputs {sorted(request.input_domains)}."
    if last_error is not None:
        message += f" A qualified execution route was rejected with: {last_error}"
    raise BiltExecutionPlanError(message) from last_error


def required_inputs_expanded(structure: ExpansionSnapshot, output: str) -> frozenset[str] | None:
    """Discover a primitive route without compiling it or requiring input domains."""
    for assignments in candidate_assignments(planning.matching_graph(structure), set(), output):
        try:
            blocks = planning.plan_blocks(assignments, set(structure.available), output)
        except BiltAnalysisError:
            continue
        return planning.required_inputs(blocks)
    return None
