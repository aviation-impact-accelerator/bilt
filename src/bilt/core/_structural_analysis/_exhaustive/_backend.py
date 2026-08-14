__all__ = ["ExhaustiveStructuralAnalyser"]

import typing

from bilt._backend import Backend, BackendCase
from bilt.core._realize_analysis._realize import RealizationFailure, realize_plan
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


class ExhaustiveStructuralAnalyser:
    """Analyser that uses an exhaustive bipartite graph search."""

    def __init__(self, constituents: typing.Collection[BiltModelConstituent]) -> None:
        self.bipartite_graph = BipartiteGraph(constituents)

    def analyse(self, available: set[str], output: str, concrete_backend: Backend) -> BackendCase:
        """Find a plan to compute the desired output, routing around steps that cannot be realized."""
        excluded: frozenset[PlanStep] = frozenset()
        last_failure: RealizationFailure | None = None
        while True:
            steps = self.bipartite_graph.search(available, output, excluded=excluded)
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
                concrete_backend,
            )
            match realised_plan:
                case BackendCase() as successful_plan:
                    return successful_plan
                case RealizationFailure(permanent=True) as failure:
                    last_failure = failure
                    self.bipartite_graph.remove_failed_plan_step(failure.failed_plan_step)
                case RealizationFailure() as failure:
                    last_failure = failure
                    excluded |= {failure.failed_plan_step}
