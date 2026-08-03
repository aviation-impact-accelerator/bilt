__all__ = ["exhaustive_structural_backend"]

import typing

from bilt._backend import Backend, BackendCase
from bilt.core._realize_analysis._realize import RealizationFailure, realize_plan
from bilt.core._structural_analysis._errors import BiltSystemEvaluationError
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._structural_analysis._plan import AnalysisPlan

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


def exhaustive_structural_backend(
    constituents: typing.Collection[BiltModelConstituent],
    available: set[str],
    output: str,
    concrete_backend: Backend,
) -> BackendCase:
    """Perform an exhaustive analysis of the model constituents to find a plan to compute the desired output."""
    bipartite_graph = BipartiteGraph(constituents)
    while True:
        steps = bipartite_graph.search(available, output, [])
        if steps is None:
            msg = f"Cannot compute {output} from available inputs {available}."
            raise BiltSystemEvaluationError(msg)

        realised_plan = realize_plan(
            AnalysisPlan(inputs=frozenset(available), output=output, steps=tuple(steps)), concrete_backend
        )
        match realised_plan:
            case BackendCase() as successful_plan:
                return successful_plan
            case RealizationFailure() as failure:
                bipartite_graph.remove_failed_plan_step(failure.failed_plan_step)
