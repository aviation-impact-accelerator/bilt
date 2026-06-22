from collections.abc import Callable
from dataclasses import dataclass

from bilt.symbolic._structural_analysis._candidates import StructuralCandidate, Variable

Isolate = Callable[[object, Variable], object | None]


@dataclass(frozen=True)
class RealizedStep:
    """A structural step with an executable implementation.

    For equations:
        implementation = isolated expression

    For transforms:
        implementation = transform object
    """

    candidate: StructuralCandidate
    implementation: object


@dataclass(frozen=True)
class ExecutionPlan:
    """Ordered executable steps."""

    steps: tuple[RealizedStep, ...]


@dataclass(frozen=True)
class RealizationFailure:
    """Reason a structurally valid candidate could not be realised."""

    candidate: StructuralCandidate
    reason: str


@dataclass(frozen=True)
class PlanningResult:
    """Result of attempting to build an execution plan.

    `plan` is None when no structural proposal could be realised.
    `structural_candidates` is returned for debugging/inspection.
    `rejected` records structurally valid proposals that failed realisation.
    """

    plan: ExecutionPlan | None
    structural_candidates: tuple[StructuralCandidate, ...]
    rejected: tuple[RealizationFailure, ...]


__all__ = [
    "ExecutionPlan",
    "Isolate",
    "PlanningResult",
    "RealizationFailure",
    "RealizedStep",
]
