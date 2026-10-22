__all__ = ["ExhaustiveStructuralAnalyser"]

import typing

from bilt._backend import BackendCase
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock
from bilt.core._structural_analysis._realize import RealizationFailure, realize_plan

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._model_constituent import BiltModelConstituent


class ExhaustiveStructuralAnalyser:
    """Analyser that uses an exhaustive bipartite graph search."""

    def __init__(self, constituents: typing.Collection[BiltModelConstituent], backend: Backend) -> None:
        self.backend = backend
        self.bipartite_graph = BipartiteGraph(constituents, backend)

    def analyse(self, request: EvaluationRequest) -> BackendCase:
        """Find a plan to satisfy the request, pruning steps that fail."""
        last_failure: RealizationFailure | None = None

        while True:
            steps = self.bipartite_graph.search(
                set(request.input_domains),
                request.output,
            )
            if steps is None:
                msg = f"Cannot compute {request.output} from available inputs {set(request.input_domains)}."
                if last_failure is None or last_failure.cause is None:
                    raise BiltAnalysisError(msg)

                msg = (
                    f"{msg} The last route tried was rejected with: "
                    f"{type(last_failure.cause).__name__}: {last_failure.cause}"
                )
                raise BiltAnalysisError(msg) from last_failure.cause

            realised_plan = realize_plan(
                AnalysisPlan(
                    request=request,
                    blocks=tuple(PlanBlock(steps=(step,)) for step in steps),
                ),
                self.backend,
            )

            match realised_plan:
                case BackendCase() as successful_plan:
                    return successful_plan
                case RealizationFailure() as failure:
                    last_failure = failure
                    self.bipartite_graph.remove_failed_plan_step(failure.failed_plan_step)

    def required_inputs_for_target(self, available: set[str], output: str) -> frozenset[str] | None:
        """Return the inputs a plan for ``output`` would actually consume, or None when no plan exists.

        This answers the structural question only -- no plan step is realized -- so a parent can
        ask a child what it needs before committing to compiling it.
        """
        try:
            steps = self.bipartite_graph.search(available, output)
        except BiltAnalysisError:
            return None

        if steps is None:
            return None

        produced = {step.output for step in steps}
        return frozenset(attribute for step in steps for attribute in step.consumed if attribute not in produced)
