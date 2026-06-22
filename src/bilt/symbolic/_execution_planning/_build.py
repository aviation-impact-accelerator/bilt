from __future__ import annotations

from bilt.execution_planning._realize import realize_structural_plan
from bilt.execution_planning._types import Isolate, PlanningResult
from bilt.structural_analysis import (
    CandidateSource,
    Case,
    StructuralBackend,
    generate_structural_candidates,
)


def build_execution_plan(
    *,
    source: CandidateSource,
    case: Case,
    isolate: Isolate,
    backend: StructuralBackend,
) -> PlanningResult:
    """
    Build an executable plan using an injected structural backend.

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

    rejected = []

    for structural_plan in backend(structural_candidates, case):
        execution_plan, failures = realize_structural_plan(
            structural_plan,
            isolate=isolate,
        )

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
