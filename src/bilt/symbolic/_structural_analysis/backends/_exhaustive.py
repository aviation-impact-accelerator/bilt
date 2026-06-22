from collections.abc import Iterable, Iterator
from itertools import product

from bilt.structural_analysis._schedule import schedule_structural_candidates
from bilt.structural_analysis._types import (
    Case,
    ProducerKey,
    StructuralCandidate,
    StructuralPlan,
)


def _group_by_producer(
    candidates: Iterable[StructuralCandidate],
) -> dict[ProducerKey, list[StructuralCandidate]]:
    """
    Group structural candidates by producer key.

    The exhaustive backend uses this to choose zero or one output orientation
    per producer.
    """

    groups: dict[ProducerKey, list[StructuralCandidate]] = {}

    for candidate in candidates:
        groups.setdefault(candidate.producer_key, []).append(candidate)

    return groups


def exhaustive_backend(
    candidates: tuple[StructuralCandidate, ...],
    case: Case,
) -> Iterator[StructuralPlan]:
    """
    Exhaustively enumerate structural producer-output choices.

    The backend chooses zero or one candidate per producer key and yields every
    structurally valid plan it can schedule from the case inputs.

    It rejects choices where:

    - two selected candidates produce the same variable
    - requested outputs are neither known inputs nor produced outputs
    - selected candidates cannot be ordered from the known inputs

    This function does not isolate equations, realise transforms, or construct
    execution plans.
    """

    groups = _group_by_producer(candidates)

    options_by_producer: list[list[StructuralCandidate | None]] = [
        [None, *options]
        for options in groups.values()
    ]

    for choice in product(*options_by_producer):
        chosen = tuple(
            candidate
            for candidate in choice
            if candidate is not None
        )

        produced_outputs = [candidate.output for candidate in chosen]
        produced_output_set = set(produced_outputs)

        # Reject duplicate produced variables.
        if len(produced_outputs) != len(produced_output_set):
            continue

        # Requested outputs must already be inputs or produced by this choice.
        if not case.outputs <= case.inputs | produced_output_set:
            continue

        structural_plan = schedule_structural_candidates(
            chosen,
            known_inputs=case.inputs,
            targets=case.outputs,
        )

        if structural_plan is not None:
            yield structural_plan


__all__ = ["exhaustive_backend"]
