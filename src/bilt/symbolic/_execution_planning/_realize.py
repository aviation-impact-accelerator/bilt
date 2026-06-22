from bilt.symbolic._execution_planning._types import (
    ExecutionPlan,
    RealizationFailure,
    RealizedStep,
)
from bilt.symbolic._structural_analysis._types import StructuralCandidate, StructuralPlan


def realize_step(
    candidate: StructuralCandidate,
    *,
    isolate: Isolate,
) -> RealizedStep | RealizationFailure:
    """
    Lower one selected structural candidate into an executable step.

    Equation candidates are isolated here:

        isolate(candidate.producer, candidate.output)

    For equation candidates, candidate.producer is the original equation tree
    carried by the structural candidate.

    Transform candidates use the transform object directly.
    """

    if candidate.kind == "equation":
        implementation = isolate(candidate.producer, candidate.output)

        if implementation is None:
            return RealizationFailure(
                candidate=candidate,
                reason=f"Could not isolate {candidate.output!r}",
            )

        return RealizedStep(
            candidate=candidate,
            implementation=implementation,
        )

    if candidate.kind == "transform":
        return RealizedStep(
            candidate=candidate,
            implementation=candidate.producer,
        )

    raise TypeError(f"Unknown candidate kind: {candidate.kind!r}")


def realize_structural_plan(
    structural_plan: StructuralPlan,
    *,
    isolate: Isolate,
) -> tuple[ExecutionPlan | None, tuple[RealizationFailure, ...]]:
    """
    Realise a structural plan.

    If any selected equation cannot be isolated, the whole structural plan is
    rejected and the caller can try the next backend proposal.
    """

    realized_steps: list[RealizedStep] = []

    for candidate in structural_plan.steps:
        result = realize_step(candidate, isolate=isolate)

        if isinstance(result, RealizationFailure):
            return None, (result,)

        realized_steps.append(result)

    return ExecutionPlan(steps=tuple(realized_steps)), tuple()


__all__ = ["realize_step", "realize_structural_plan"]
