from __future__ import annotations

from dataclasses import dataclass

from bilt.structural_analysis._types import ProducerKey, Variable


@dataclass(frozen=True)
class EquationSpec:
    """
    Equation available for structural analysis.

    `producer` is intentionally opaque to this layer. In the real system this
    can be the original symbolic equation tree.
    """

    key: ProducerKey
    producer: object
    variables: frozenset[Variable]


@dataclass(frozen=True)
class TransformSpec:
    """
    Transform available for structural analysis.

    `producer` is intentionally opaque to this layer. In the real system this
    can be the transform object.
    """

    key: ProducerKey
    producer: object
    inputs: frozenset[Variable]
    output: Variable


@dataclass(frozen=True)
class CandidateSource:
    """
    Normalised source records from which structural candidates can be generated.
    """

    equations: tuple[EquationSpec, ...]
    transforms: tuple[TransformSpec, ...] = ()


__all__ = [
    "CandidateSource",
    "EquationSpec",
    "TransformSpec",
]
