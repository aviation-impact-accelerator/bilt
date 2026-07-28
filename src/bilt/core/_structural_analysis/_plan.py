from __future__ import annotations

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True)
class PlanStep:
    """A single step in an analysis plan.

    Pairs a model constituent with the specific output it is solving for.
    For transforms, ``output`` equals the transform's ``__bilt_name__``.
    For equations, ``output`` is whichever parameter is being isolated.

    This is intentionally decoupled from the actual isolation or system evaluation logic which takes place in _realize_analysis.
    We are only looking to determine if the path is valid rather than if the steps can actually be used.
    """

    operation: BiltModelConstituent
    output: str


@dataclasses.dataclass(frozen=True)
class AnalysisPlan:
    inputs: frozenset[str]
    steps: tuple[PlanStep, ...]
    output: str
