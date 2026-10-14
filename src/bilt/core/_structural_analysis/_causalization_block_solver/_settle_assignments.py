__all__ = ["settle_assignments"]

import typing

from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent


def settle_assignments(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: collections.abc.Set[str],
    backend: Backend,
) -> tuple[tuple[PlanStep, ...], frozenset[str]]:
    """Select child routes for candidate assignments.

    Supplied attributes and matched outputs are candidate inputs.
    Dependencies and realisability are checked later.

    Rejected outputs apply to this candidate input set, rather than
    establishing that the subsystem can never produce those outputs.
    """
    candidate_inputs = set(available) | set(assignments)
    steps: list[PlanStep] = []
    rejected: set[str] = set()

    for output, constituent in assignments.items():
        child_inputs = None

        if isinstance(constituent, SubSystemConstituent):
            candidate_ports = set(constituent.__bilt_parameters__) & candidate_inputs
            child_inputs = constituent.required_inputs_for(
                output,
                backend,
                within=candidate_ports,
            )
            if child_inputs is None:
                rejected.add(output)
                continue

        steps.append(
            PlanStep(
                operation=constituent,
                output=output,
                child_inputs=child_inputs,
            )
        )

    return tuple(steps), frozenset(rejected)
