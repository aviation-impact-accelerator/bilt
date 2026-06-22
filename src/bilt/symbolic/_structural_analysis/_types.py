from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Variable = str
ProducerKey = str
ProducerKind = Literal["equation", "transform"]


@dataclass(frozen=True)
class Case:
    """
    Minimal structural-analysis request.

    `inputs` are variables already available.
    `outputs` are variables that must be available by the end of the structural
    plan.
    """

    inputs: frozenset[Variable]
    outputs: frozenset[Variable]


@dataclass(frozen=True)
class StructuralCandidate:
    """
    A possible producer-output edge.

    The structural-analysis layer uses `producer_key`, `output`, and `inputs`.

    `kind` and `producer` are carried opaquely so a later execution-planning
    layer can realise selected candidates without doing another lookup.
    """

    producer_key: ProducerKey
    kind: ProducerKind
    producer: object
    output: Variable
    inputs: frozenset[Variable]


@dataclass(frozen=True)
class StructuralPlan:
    """
    Ordered structural plan emitted by a structural backend.

    Steps are structural candidates only. No equation isolation or execution
    lowering has happened at this layer.
    """

    steps: tuple[StructuralCandidate, ...]


__all__ = [
    "Case",
    "ProducerKey",
    "ProducerKind",
    "StructuralCandidate",
    "StructuralPlan",
    "Variable",
]
