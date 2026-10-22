__all__ = ["CausalizationStructuralAnalyzer"]

import typing

from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver import _planning as planning
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._coupling import unavoidable_coupled_outputs
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
    FullIncidenceGraph,
)
from bilt.core._structural_analysis._causalization_block_solver._matching import (
    build_matching_graph,
    prefer_structural_producers,
)
from bilt.core._structural_analysis._causalization_block_solver._route_cache import (
    invalidate_route_cache,
    route_cache_scope,
)
from bilt.core._structural_analysis._causalization_block_solver._structural_cache import structural_cache_scope
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import (
    validate_request_structure,
)
from bilt.core._structural_analysis._plan import AnalysisPlan
from bilt.core._structural_analysis._realize import (
    RealizationFailure,
    realize_plan,
)
from bilt.core._sub_system import SubSystemConstituent

if typing.TYPE_CHECKING:
    import rustworkx as rx

    from bilt._backend import Backend, BackendCase
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import ExpansionSnapshot
    from bilt.core._structural_analysis._plan import PlanStep


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
        """Find and realise a plan, sharing request caches across nested analysis."""
        with structural_cache_scope(), route_cache_scope():
            return self._analyse(request)

    def _analyse(self, request: EvaluationRequest) -> BackendCase:
        """Search within the current request's cache scopes."""
        available = set(request.input_domains)
        structure = validate_request_structure(
            self.full_incidence_graph,
            available,
            request.output,
        )
        if structure is not None and (
            coupled := unavoidable_coupled_outputs(structure.well_determined, request.output)
        ):
            msg = (
                f"Cannot compute {request.output} from available inputs {sorted(available)}. "
                f"Execution of coupled blocks is not yet supported by this analyser. "
                f"Required coupled computations: {', '.join(sorted(coupled))}."
            )
            raise BiltExecutionPlanError(msg)

        graph = build_matching_graph(self.full_incidence_graph, available)
        self._prefer_child_producers(graph, available, request.output, structure.snapshot if structure else None)
        last_error: Exception | None = None

        for assignments in search.candidate_assignments(
            graph, self._failures, output=None if request.output in available else request.output
        ):
            attempted_routes: set[tuple[tuple[str, frozenset[str] | None], ...]] = set()

            while True:
                try:
                    steps = planning.plan_steps(
                        assignments,
                        available,
                        request.output,
                        backend=self.backend,
                    )
                except BiltAnalysisError as error:
                    if last_error is None:
                        last_error = error
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

                # Child analysis may have learned failures, too. Invalidate all
                # dependent parent answers before the next route query.
                invalidate_route_cache()
                last_error = result.cause
                failed_step = result.failed_plan_step

                if isinstance(failed_step.operation, SubSystemConstituent):
                    # Child analysis may have rejected an internal step.
                    # Query it again before abandoning this assignment.
                    continue

                self._failures[failed_step] = result
                break

        msg = f"Cannot compute {request.output} from available inputs {available}."
        if last_error is not None:
            msg = f"{msg} An execution route was rejected with: {type(last_error).__name__}: {last_error}"
            raise BiltExecutionPlanError(msg) from last_error

        raise BiltExecutionPlanError(msg)

    def required_inputs_for_target(
        self,
        available: set[str],
        output: str,
    ) -> frozenset[str] | None:
        """Reuse an execution-route query within the current nested analysis."""
        with structural_cache_scope(), route_cache_scope() as cache:
            key = self, frozenset(available), output
            if key not in cache:
                cache[key] = self._required_inputs_for_target(available, output)
            return cache[key]

    def _required_inputs_for_target(self, available: set[str], output: str) -> frozenset[str] | None:
        """Search for route inputs without realising operations."""
        graph = build_matching_graph(self.full_incidence_graph, available)
        self._prefer_child_producers(graph, available, output)

        for assignments in search.candidate_assignments(
            graph, self._failures, output=None if output in available else output
        ):
            try:
                steps = planning.plan_steps(assignments, available, output, backend=self.backend)
            except BiltAnalysisError:
                continue

            if output in available:
                return frozenset({output})

            produced = {step.output for step in steps}
            return frozenset(attribute for step in steps for attribute in step.consumed if attribute not in produced)

        return None

    def _prefer_child_producers(
        self,
        graph: rx.PyGraph,
        available: set[str],
        output: str,
        structure: ExpansionSnapshot | None = None,
    ) -> None:
        """Order competing child producers without expanding simple chains again."""
        if output in available:
            return
        child_count = sum(
            isinstance(self.full_incidence_graph.graph[node], SubSystemConstituent)
            for node in self.full_incidence_graph.constituent_nodes
        )
        if child_count <= 1:
            return
        if structure is None:
            try:
                expansion = StructuralExpansion(self.full_incidence_graph, available, output)
                expansion.collect()
            except BiltAnalysisError:
                # Route queries are not definitive validation. A failed hint
                # must leave their existing execution search available.
                return
            structure = expansion.snapshot()
        prefer_structural_producers(graph, structure)
