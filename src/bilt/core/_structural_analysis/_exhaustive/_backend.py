__all__ = ["ExhaustiveStructuralAnalyser"]

import typing

from bilt._backend import BackendCase
from bilt.core._realize_analysis._realize import RealizationFailure, realize_plan
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._structural_analysis._plan import AnalysisPlan

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent


class ExhaustiveStructuralAnalyser:
    """Analyser that uses an exhaustive bipartite graph search."""

    def __init__(self, constituents: typing.Collection[BiltModelConstituent], backend: Backend) -> None:
        self.backend = backend
        self.bipartite_graph = BipartiteGraph(constituents, backend)

    def analyse(self, available: set[str], output: str) -> BackendCase:
        """Find a plan to compute the desired output, pruning steps that fail."""
        last_failure: RealizationFailure | None = None
        while True:
            steps = self.bipartite_graph.search(available, output)
            if steps is None:
                msg = f"Cannot compute {output} from available inputs {available}."
                if last_failure is None or last_failure.cause is None:
                    raise BiltSystemAnalysisError(msg)
                msg = (
                    f"{msg} The last route tried was rejected with: "
                    f"{type(last_failure.cause).__name__}: {last_failure.cause}"
                )
                raise BiltSystemAnalysisError(msg) from last_failure.cause

            realised_plan = realize_plan(
                AnalysisPlan(inputs=frozenset(available), output=output, steps=tuple(steps)),
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
        except BiltSystemAnalysisError:
            return None

        if steps is None:
            return None

        produced = {step.output for step in steps}
        required: set[str] = set()
        for step in steps:
            consumed = self.bipartite_graph.required_inputs_for_step(step) or frozenset()
            required.update(consumed - produced)

        return frozenset(required)
