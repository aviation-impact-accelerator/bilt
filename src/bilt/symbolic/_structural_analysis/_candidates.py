import typing

if typing.TYPE_CHECKING:
    from bilt.symbolic._structural_analysis._sources import CandidateSource
from bilt.symbolic._structural_analysis._types import StructuralCandidate, Variable


def generate_structural_candidates(
    source: CandidateSource,
    *,
    known_inputs: frozenset[Variable],
) -> tuple[StructuralCandidate, ...]:
    """Generate all structural candidates from the available producers.

    For each equation, generate one candidate per possible output variable.
    Equation candidates whose output is already known are skipped.

    For transforms, generate one candidate from the transform's fixed inputs and
    output. Transform outputs that are already known are skipped.

    No scheduling, equation isolation, or realisation happens here.
    """
    candidates: list[StructuralCandidate] = []

    for equation in source.equations:
        for output in equation.variables:
            if output in known_inputs:
                continue

            candidates.append(
                StructuralCandidate(
                    producer_key=equation.key,
                    kind="equation",
                    producer=equation.producer,
                    output=output,
                    inputs=frozenset(equation.variables - {output}),
                )
            )

    for transform in source.transforms:
        if transform.output in known_inputs:
            continue

        candidates.append(
            StructuralCandidate(
                producer_key=transform.key,
                kind="transform",
                producer=transform.producer,
                output=transform.output,
                inputs=transform.inputs,
            )
        )

    return tuple(candidates)


__all__ = ["generate_structural_candidates"]
