__all__ = ["SCCStructuralAnalyser"]

import typing

from bilt._backend import Backend, BackendCase
from bilt.core._realize_analysis._realize import (
    RealizationFailure,
    realize_plan,
)
from bilt.core._structural_analysis._errors import (
    BiltSystemAnalysisError,
)
from bilt.core._structural_analysis._scc._bipartite import (
    BipartiteGraph,
)
from bilt.core._structural_analysis._scc._dependencies import (
    required_inputs_for_plan,
)

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import (
        AnalysisPlan,
        PlanStep,
    )


class SCCStructuralAnalyser:
    """Analyser using structural matching and SCC decomposition."""

    def __init__(
        self,
        constituents: typing.Collection[BiltModelConstituent],
    ) -> None:
        self.bipartite_graph = BipartiteGraph(constituents)

    def analyse(
        self,
        available: set[str],
        output: str,
        concrete_backend: Backend,
    ) -> BackendCase:
        """Find and realise a plan to compute the desired output."""
        excluded: frozenset[PlanStep] = frozenset()

        while True:
            plan = self.bipartite_graph.find_plan(
                available,
                output,
                excluded=excluded,
            )

            if plan is None:
                msg = f"Cannot compute {output} from available inputs {available}."
                raise BiltSystemAnalysisError(msg)

            match realize_plan(
                plan,
                concrete_backend,
            ):
                case BackendCase() as successful_plan:
                    return successful_plan

                case RealizationFailure() as failure:
                    excluded |= {failure.failed_plan_step}

    def plan_for_target(
        self,
        available: set[str],
        output: str,
    ) -> AnalysisPlan | None:
        """Return a structural plan for a target, if one exists."""
        try:
            return self.bipartite_graph.find_plan(
                available,
                output,
            )
        except BiltSystemAnalysisError:
            return None

    def required_inputs_for_target(
        self,
        available: set[str],
        output: str,
    ) -> frozenset[str] | None:
        """Return the inputs required to structurally produce a target."""
        plan = self.plan_for_target(
            available,
            output,
        )

        if plan is None:
            return None

        return required_inputs_for_plan(plan)
