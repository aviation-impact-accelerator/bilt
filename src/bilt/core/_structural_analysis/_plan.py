from __future__ import annotations

__all__ = [
    "AnalysisPlan",
    "CoupledBlock",
    "PlanItem",
    "PlanStep",
    "SubSystemPlan",
]

import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True)
class PlanStep:
    """A single step in an analysis plan."""

    operation: BiltModelConstituent
    output: str
    sub_system_plan: SubSystemPlan | None = dataclasses.field(
        default=None,
        compare=False,
        hash=False,
    )


@dataclasses.dataclass(frozen=True)
class CoupledBlock:
    """A set of plan steps that must be solved together."""

    steps: tuple[PlanStep, ...]

    @property
    def outputs(self) -> frozenset[str]:
        """Return the outputs produced by the block."""
        return frozenset(step.output for step in self.steps)


type PlanItem = PlanStep | CoupledBlock


@dataclasses.dataclass(frozen=True)
class AnalysisPlan:
    """A structural plan to compute an output from available inputs."""

    inputs: frozenset[str]
    steps: tuple[PlanItem, ...]
    output: str


@dataclasses.dataclass(frozen=True)
class SubSystemPlan:
    """A selected structural route through a child system."""

    plan: AnalysisPlan
    inputs: frozenset[str]
