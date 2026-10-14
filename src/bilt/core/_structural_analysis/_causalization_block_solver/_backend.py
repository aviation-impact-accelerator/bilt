__all__ = ["CausalizationStructuralAnalyzer"]

import typing

import rustworkx as rx

from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver._blocks import (
    ordered_blocks,
)
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import (
    build_dependency_graph,
)
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._matching import (
    build_matching_graph,
    find_matching,
)
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import (
    settle_assignments,
)
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import (
    validate_request_structure,
)
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
from bilt.core._structural_analysis._realize import (
    RealizationFailure,
    realize_plan,
)
from bilt.core._sub_system import SubSystemConstituent

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend import Backend, BackendCase
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._model_constituent import BiltModelConstituent


class CausalizationStructuralAnalyzer:
    """Construct plans using matching and dependency decomposition."""

    def __init__(
        self,
        constituents: typing.Collection[BiltModelConstituent],
        backend: Backend,
    ) -> None:
        self.backend = backend
        self.full_incidence_graph = FullIncidenceGraph(constituents)
        self._failures: dict[PlanStep, RealizationFailure] = {}

    def analyse(self, request: EvaluationRequest) -> BackendCase:
        """Find and realise a plan, retrying unsuccessful assignments."""
        available = set(request.input_domains)
        validate_request_structure(
            self.full_incidence_graph,
            available,
            request.output,
            self.backend,
        )
        graph = build_matching_graph(self.full_incidence_graph, available)
        last_error: Exception | None = None

        for assignments in _candidate_assignments(graph, self._failures):
            attempted_routes: set[tuple[tuple[str, frozenset[str] | None], ...]] = set()

            while True:
                try:
                    steps = self._plan_steps(
                        assignments,
                        available,
                        request.output,
                    )
                except BiltAnalysisError as error:
                    last_error = error
                    break

                if steps is None:
                    break

                route = tuple((step.output, step.child_inputs) for step in steps)
                if route in attempted_routes:
                    break
                attempted_routes.add(route)

                result = realize_plan(
                    AnalysisPlan(request=request, steps=steps),
                    self.backend,
                )
                if not isinstance(result, RealizationFailure):
                    return result

                last_error = result.cause
                failed_step = result.failed_plan_step

                if isinstance(failed_step.operation, SubSystemConstituent):
                    # Child analysis may have rejected an internal step.
                    # Query it again before abandoning this assignment.
                    continue

                self._failures[failed_step] = result
                break

        msg = f"Cannot compute {request.output} from available inputs {available}."
        if isinstance(last_error, (BiltUnderconstrainedError, BiltOverconstrainedError)):
            raise last_error
        if last_error is not None:
            msg = f"{msg} The last route tried was rejected with: {type(last_error).__name__}: {last_error}"
            raise BiltAnalysisError(msg) from last_error

        raise BiltUnderconstrainedError(msg)

    def required_inputs_for_target(
        self,
        available: set[str],
        output: str,
    ) -> frozenset[str] | None:
        """Return the inputs used by a structural plan, without realising it."""
        graph = build_matching_graph(self.full_incidence_graph, available)

        for assignments in _candidate_assignments(graph, self._failures):
            try:
                steps = self._plan_steps(assignments, available, output)
            except BiltAnalysisError:
                continue

            if steps is None:
                continue

            if output in available:
                return frozenset({output})

            produced = {step.output for step in steps}
            return frozenset(attribute for step in steps for attribute in step.consumed if attribute not in produced)

        return None

    def _plan_steps(
        self,
        assignments: collections.abc.Mapping[str, BiltModelConstituent],
        available: set[str],
        output: str,
    ) -> tuple[PlanStep, ...] | None:
        """Select the target's dependencies and order an acyclic plan."""
        if output in available:
            return ()

        if output not in assignments:
            return None

        settled, _ = settle_assignments(
            assignments,
            available,
            self.backend,
        )
        by_output = {step.output: step for step in settled}
        selected: dict[str, PlanStep] = {}
        pending = [output]

        while pending:
            attribute = pending.pop()

            if attribute in available or attribute in selected:
                continue

            step = by_output.get(attribute)
            if step is None:
                return None

            selected[attribute] = step
            pending.extend(step.consumed)

        graph, unresolved = build_dependency_graph(
            tuple(selected.values()),
            available,
        )
        if unresolved:
            return None

        blocks = ordered_blocks(graph)
        if not rx.is_directed_acyclic_graph(graph):
            outputs = ", ".join(sorted(selected))
            msg = (
                f"Execution of coupled blocks is not yet supported by this analyser. Required computations: {outputs}."
            )
            raise BiltAnalysisError(msg)

        return tuple(step for block in blocks for step in block)


def _edge_step(graph: rx.PyGraph, first: int, second: int) -> PlanStep:
    """Identify the computation represented by a matching edge."""
    if isinstance(graph[first], str):
        attribute_node, constituent_node = first, second
    else:
        constituent_node, attribute_node = first, second

    candidate = graph[constituent_node]
    if isinstance(candidate, PlanStep):
        return candidate

    return PlanStep(
        operation=typing.cast("BiltModelConstituent", candidate),
        output=typing.cast("str", graph[attribute_node]),
    )


def _candidate_assignments(
    graph: rx.PyGraph,
    failures: collections.abc.Mapping[PlanStep, RealizationFailure],
) -> collections.abc.Iterator[dict[str, BiltModelConstituent]]:
    """Try matchings while excluding failed or unsuccessful choices.

    Failed realisations exclude individual assignments. Other rejected
    candidates branch by removing selected matching edges.
    """
    pending = [graph]
    visited: set[frozenset[tuple[int, int]]] = set()

    while pending:
        candidate_graph = pending.pop()

        for first, second in list(candidate_graph.edge_list()):
            if _edge_step(candidate_graph, first, second) in failures:
                candidate_graph.remove_edge(first, second)

        edges = list(candidate_graph.edge_list())
        signature = frozenset((min(first, second), max(first, second)) for first, second in edges)
        if signature in visited:
            continue
        visited.add(signature)

        assignments = find_matching(candidate_graph)
        yield assignments

        # Realisation may have added failures while the generator yielded.
        # Retry this graph with those assignments excluded first.
        if any(_edge_step(candidate_graph, first, second) in failures for first, second in edges):
            pending.append(candidate_graph)
            continue

        for first, second in edges:
            step = _edge_step(candidate_graph, first, second)
            if assignments.get(step.output) is not step.operation:
                continue

            alternative = candidate_graph.copy()
            alternative.remove_edge(first, second)
            pending.append(alternative)
