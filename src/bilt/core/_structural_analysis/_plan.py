from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True)
class PlanStep:
    """A single step in an analysis plan.

    Pairs a single model constituent with the output it is solving for and inputs needed for it.
    For transforms, ``output`` equals the transform's ``__bilt_name__``.
    For equations, ``output`` is whichever parameter is being isolated.

    This is intentionally decoupled from the actual isolation or system evaluation logic which takes place in _realize_analysis.
    We are only looking to determine if the path is valid through the system graph rather than if the steps can actually be performed.
    """

    inputs: frozenset[str]
    operation: BiltModelConstituent
    output: str


@dataclasses.dataclass(frozen=True)
class AnalysisPlan:
    """A plan through a complete system graph to compute a desired output from available inputs."""

    inputs: frozenset[str]
    steps: tuple[PlanStep, ...]
    output: str
