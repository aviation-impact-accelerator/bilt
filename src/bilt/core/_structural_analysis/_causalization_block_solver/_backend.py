__all__ = ["CausalizationStructuralAnalyzer"]

import typing

from bilt.core._errors import BiltAnalysisError, BiltEquationDomainRequirementError
from bilt.core._structural_analysis._causalization_block_solver import _planning as planning
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._coupling import unavoidable_coupled_outputs
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._expanded_search import (
    analyse_expanded,
    required_inputs_expanded,
)
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
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
from bilt.core._structural_analysis._realize import (
    BlockRealizationFailure,
    BlockRealizationKey,
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
    from bilt.core._structural_analysis._causalization_block_solver._native_prototype import NativePlanner
    from bilt.core._structural_analysis._causalization_block_solver._structural_validation import ValidatedStructure
    from bilt.symbolic._domains import Domain


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
        self._experimental_native_initialized = False
        self._experimental_native: NativePlanner | None = None
        # A successful matching is only a starting point for later targets.
        # Structural validation and realisation still run for every request.
        self._last_matching: tuple[frozenset[tuple[str, Domain]], dict[str, BiltModelConstituent]] | None = None

    def analyse(self, request: EvaluationRequest) -> BackendCase:
        """Find and realise a plan, sharing request caches across nested analysis."""
        from ._native_prototype import attempt  # noqa: PLC0415

        with structural_cache_scope(), route_cache_scope():
            native = attempt(self, request, (request.output,))
            if isinstance(native, (RealizationFailure, BlockRealizationFailure)):
                self._retry_child(native)
                return self._analyse(request, initial_error=native.cause)
            if native is not None:
                return native
            return self._analyse(request)

    def _analyse(self, request: EvaluationRequest, *, initial_error: Exception | None = None) -> BackendCase:
        """Search within the current request's cache scopes."""
        available = set(request.input_domains)
        structure = validate_request_structure(
            self.full_incidence_graph,
            available,
            request.output,
        )
        expanded = self._preflight(request, structure)
        if expanded is not None:
            return expanded

        signature = frozenset(request.input_domains.items())
        last_error = initial_error
        block_failures: dict[BlockRealizationKey, BlockRealizationFailure] = {}

        for assignments in self._matching_candidates(request, available, structure, signature):
            attempted_routes: set[tuple[tuple[str, frozenset[str] | None], ...]] = set()

            while True:
                try:
                    blocks = planning.plan_blocks(
                        assignments,
                        available,
                        request.output,
                        backend=self.backend,
                    )
                except BiltAnalysisError as error:
                    if last_error is None:
                        last_error = error
                    break

                route = tuple((step.output, step.child_inputs) for block in blocks for step in block.steps)
                if route in attempted_routes:
                    break
                attempted_routes.add(route)

                result = realize_plan(
                    AnalysisPlan(request=request, blocks=blocks),
                    self.backend,
                    block_failures=block_failures,
                )
                if not isinstance(result, (RealizationFailure, BlockRealizationFailure)):
                    self._last_matching = signature, assignments.copy()
                    return result

                # Invalidate dependent parent answers before the next route query.
                invalidate_route_cache()
                last_error = result.cause
                if self._retry_child(result):
                    continue
                break

        msg = f"Cannot compute {request.output} from available inputs {available}."
        if last_error is not None:
            msg = f"{msg} An execution route was rejected with: {type(last_error).__name__}: {last_error}"
            raise BiltExecutionPlanError(msg) from last_error

        raise BiltExecutionPlanError(msg)

    def _matching_candidates(
        self,
        request: EvaluationRequest,
        available: set[str],
        structure: ValidatedStructure | None,
        signature: frozenset[tuple[str, Domain]],
    ) -> typing.Iterator[dict[str, BiltModelConstituent]]:
        """Try the last matching before rebuilding the graph for this target.

        Reusing an assignment never bypasses structural validation or plan
        realisation. A failed or incompatible seed resumes the normal search.
        Keep only one successful seed to bound retained state.
        """
        last = self._last_matching
        if last is not None and last[0] == signature:
            assignments = last[1]
            if not self._failures or not any(
                PlanStep(operation=operation, output=name) in self._failures for name, operation in assignments.items()
            ):
                yield assignments

        graph = build_matching_graph(self.full_incidence_graph, available)
        self._prefer_child_producers(graph, available, request.output, structure.snapshot if structure else None)
        yield from search.candidate_assignments(
            graph, self._failures, output=None if request.output in available else request.output
        )

    def analyse_many(self, request: EvaluationRequest, outputs: tuple[str, ...]) -> BackendCase:
        """Choose one matching and realise the union of target dependencies.

        Each target is validated against the *original* supplied-input boundary.
        This opt-in path supports ordinary parent/child plans; callers may
        fall back to separate evaluation when no compatible joint route exists.
        """
        targets = tuple(dict.fromkeys(outputs))
        if not targets or request.output not in targets:
            msg = "Multi-output planning requires a nonempty target set containing the primary output."
            raise ValueError(msg)
        if len(targets) == 1:
            return self.analyse(request)
        from ._native_prototype import attempt  # noqa: PLC0415

        with structural_cache_scope(), route_cache_scope():
            native = attempt(self, request, targets)
            if native is not None and not isinstance(native, (RealizationFailure, BlockRealizationFailure)):
                return native
            return self._analyse_many(request, targets)

    def _analyse_many(self, request: EvaluationRequest, targets: tuple[str, ...]) -> BackendCase:
        available = set(request.input_domains)
        for target in targets:
            structure = validate_request_structure(self.full_incidence_graph, available, target)
            if structure is None or unavoidable_coupled_outputs(
                structure.well_determined, target, allow_local_linear=True
            ):
                msg = f"Multi-output planning cannot jointly plan the coupled target {target!r}."
                raise BiltExecutionPlanError(msg)

        graph = build_matching_graph(self.full_incidence_graph, available)
        self._prefer_child_producers(graph, available, targets[0])
        block_failures: dict[BlockRealizationKey, BlockRealizationFailure] = {}
        last_error: Exception | None = None
        # Bound candidate exploration; the existing single-output route
        # remains the correctness-preserving fallback for unsupported sets.
        for index, assignments in enumerate(search.candidate_assignments(graph, self._failures, output=None)):
            if index >= 24:  # noqa: PLR2004 - bound joint matching search
                break
            try:
                blocks = planning.plan_blocks_many(assignments, available, targets, backend=self.backend)
            except BiltAnalysisError as error:
                last_error = error
                continue
            result = realize_plan(
                AnalysisPlan(request=request, blocks=blocks),
                self.backend,
                block_failures=block_failures,
            )
            if isinstance(result, (RealizationFailure, BlockRealizationFailure)):
                last_error = result.cause
                invalidate_route_cache()
                # A joint formulation can fail even when individual routes work.
                # Do not blacklist its constituents for later requests.
                continue
            produced = {item.port for item in result.outputs}
            if not set(targets) <= (produced | available):
                last_error = BiltExecutionPlanError("Joint plan omitted a requested output.")
                continue
            return result

        msg = f"Cannot jointly plan {targets!r} from supplied inputs {sorted(available)}."
        raise BiltExecutionPlanError(msg) from last_error

    def _preflight(self, request: EvaluationRequest, structure: ValidatedStructure | None) -> BackendCase | None:
        """Keep local requests on their existing path; expand execution only for cross-boundary coupling."""
        if structure is None or not (
            coupled := unavoidable_coupled_outputs(structure.well_determined, request.output, allow_local_linear=True)
        ):
            return None
        if not unavoidable_coupled_outputs(
            structure.well_determined, request.output, allow_local_linear=True, allow_cross_boundary_linear=True
        ):
            return analyse_expanded(request, structure.well_determined, self.backend)
        msg = (
            f"Cannot compute {request.output} from available inputs {sorted(request.input_domains)}. "
            f"Execution of coupled blocks is not yet supported by this analyser. "
            f"Required coupled computations: {', '.join(sorted(coupled))}."
        )
        raise BiltExecutionPlanError(msg)

    def _retry_child(self, failure: RealizationFailure | BlockRealizationFailure) -> bool:
        """Retry child route discovery, or let matching advance to another candidate."""
        if isinstance(failure, BlockRealizationFailure):
            # The request cache rejects only this block formulation and domains.
            # No individual constituent assignment has been proven invalid.
            return False
        step = failure.failed_plan_step
        if isinstance(failure.cause, BiltEquationDomainRequirementError):
            # Domain mismatch is specific to this request. Try another
            # candidate route, but don't blacklist this equation permanently.
            return False
        if isinstance(step.operation, SubSystemConstituent):
            # Child analysis may have rejected an internal step. Query it again
            # before abandoning this assignment.
            return True
        self._failures[step] = failure
        return False

    def required_inputs_for_target(
        self,
        available: set[str],
        output: str,
    ) -> frozenset[str] | None:
        """Reuse an execution-route query within the current nested analysis."""
        from ._native_prototype import attempt_required_inputs  # noqa: PLC0415

        with structural_cache_scope(), route_cache_scope() as cache:
            key = self, frozenset(available), output
            if key not in cache:
                native = attempt_required_inputs(self, available, output)
                cache[key] = native if native is not None else self._required_inputs_for_target(available, output)
            return cache[key]

    def _required_inputs_for_target(self, available: set[str], output: str) -> frozenset[str] | None:
        """Search for route inputs without realising operations."""
        graph = build_matching_graph(self.full_incidence_graph, available)
        self._prefer_child_producers(graph, available, output)
        needs_expanded = False

        for assignments in search.candidate_assignments(
            graph, self._failures, output=None if output in available else output
        ):
            try:
                blocks = planning.plan_blocks(assignments, available, output, backend=self.backend)
            except BiltAnalysisError as error:
                needs_expanded |= isinstance(error.__cause__, BiltLinearBlockError)
                continue

            if output in available:
                return frozenset({output})

            produced = {step.output for block in blocks for step in block.steps}
            return frozenset(
                attribute
                for block in blocks
                for step in block.steps
                for attribute in step.consumed
                if attribute not in produced
            )

        return self._expanded_required_inputs(available, output) if needs_expanded else None

    def _expanded_required_inputs(self, available: set[str], output: str) -> frozenset[str] | None:
        if not any(
            isinstance(self.full_incidence_graph.graph[node], SubSystemConstituent)
            for node in self.full_incidence_graph.constituent_nodes
        ):
            return None
        try:
            structure = validate_request_structure(self.full_incidence_graph, available, output)
        except BiltAnalysisError:
            return None
        if structure is None or not unavoidable_coupled_outputs(
            structure.well_determined, output, allow_local_linear=True
        ):
            return None
        if not unavoidable_coupled_outputs(
            structure.well_determined, output, allow_local_linear=True, allow_cross_boundary_linear=True
        ):
            return required_inputs_expanded(structure.well_determined, output)
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
