from __future__ import annotations

from collections.abc import Iterable

from bilt.structural_analysis._types import StructuralCandidate, StructuralPlan, Variable


def schedule_structural_candidates(
    candidates: Iterable[StructuralCandidate],
    *,
    known_inputs: frozenset[Variable],
    targets: frozenset[Variable],
) -> StructuralPlan | None:
    """
    Try to order selected structural candidates from known inputs.

    This is structural only. It does not realise producers and does not isolate
    equations.

    A candidate can be scheduled when all of its required inputs are already
    available. Once scheduled, its output becomes available for later steps.
    """

    available = set(known_inputs)
    remaining = list(candidates)
    ordered: list[StructuralCandidate] = []

    while remaining:
        next_candidate = None

        for candidate in remaining:
            if candidate.inputs <= available:
                next_candidate = candidate
                break

        if next_candidate is None:
            return None

        ordered.append(next_candidate)
        available.add(next_candidate.output)
        remaining.remove(next_candidate)

    if not targets <= available:
        return None

    return StructuralPlan(steps=tuple(ordered))


__all__ = ["schedule_structural_candidates"]
