import typing

from bilt.symbolic._execution_planning._realize import realize_structural_plan
from bilt.symbolic._execution_planning._types import PlanningResult
from bilt.symbolic._structural_analysis._candidates import (
    generate_structural_candidates,
)
from bilt.symbolic._structural_analysis._protocol import (
    StructuralBackend,
)

if typing.TYPE_CHECKING:
    from bilt.symbolic._structural_analysis._protocol import (
        StructuralBackend,
    )
    from bilt.symbolic._structural_analysis._sources import CandidateSource
    from bilt.symbolic._structural_analysis._types import Case, Isolate


def build_execution_plan(
    *,
    source: CandidateSource,
    case: Case,
    isolate: Isolate,
    backend: StructuralBackend,
) -> PlanningResult:
    """Build an executable plan using an injected structural backend.

    Flow:
        CandidateSource
          -> StructuralCandidate pool
          -> backend yields StructuralPlan proposals
          -> selected StructuralPlan is realised using isolate()
          -> ExecutionPlan

    Concrete backends live in bilt.structural_analysis.backends.
    """
    structural_candidates = generate_structural_candidates(
        source,
        known_inputs=case.inputs,
    )

    rejected: list[typing.Any] = []

    for structural_plan in backend(structural_candidates, case):
        execution_plan, failures = realize_structural_plan(
            structural_plan,
            isolate=isolate,
        )

        failures = typing.cast("tuple[typing.Any, ...]", failures)

        if execution_plan is not None:
            return PlanningResult(
                plan=execution_plan,
                structural_candidates=structural_candidates,
                rejected=tuple(rejected),
            )

        rejected.extend(failures)

    return PlanningResult(
        plan=None,
        structural_candidates=structural_candidates,
        rejected=tuple(rejected),
    )


__all__ = ["build_execution_plan"]
