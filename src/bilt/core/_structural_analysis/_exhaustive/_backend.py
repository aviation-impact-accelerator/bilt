__all__ = ["ExhaustiveStructuralAnalyser"]

import typing

from bilt._backend import Backend, BackendCase
from bilt.core._realize_analysis._realize import RealizationFailure, realize_plan
from bilt.core._structural_analysis._errors import BiltSystemAnalysisError
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph, required_inputs_for_target
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


class ExhaustiveStructuralAnalyser:
    """Analyser that uses an exhaustive bipartite graph search."""

    def __init__(self, constituents: typing.Collection[BiltModelConstituent]) -> None:
        self.bipartite_graph = BipartiteGraph(constituents)

    def analyse(self, available: set[str], output: str, concrete_backend: Backend) -> BackendCase:
        """Perform analysis to find a plan to compute the desired output."""
        excluded: frozenset[PlanStep] = frozenset()
        while True:
            steps = self.bipartite_graph.search(available, output, excluded=excluded)
            if steps is None:
                msg = f"Cannot compute {output} from available inputs {available}."
                raise BiltSystemAnalysisError(msg)

            realised_plan = realize_plan(
                AnalysisPlan(inputs=frozenset(available), output=output, steps=tuple(steps)),
                concrete_backend,
            )
            match realised_plan:
                case BackendCase() as successful_plan:
                    return successful_plan
                case RealizationFailure() as failure:
                    excluded |= {failure.failed_plan_step}

    def required_inputs_for_target(self, available: set[str], output: str) -> frozenset[str] | None:
        """Return the inputs required to structurally produce a given output."""
        try:
            steps = self.bipartite_graph.search(available, output)
        except BiltSystemAnalysisError:
            return None

        if steps is None:
            return None

        produced = {step.output for step in steps}
        required: set[str] = set()
        for step in steps:
            consumed = required_inputs_for_target(step) or frozenset()
            required.update(consumed - produced)

        return frozenset(required)
