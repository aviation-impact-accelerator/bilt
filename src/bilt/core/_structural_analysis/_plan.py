__all__ = ["AnalysisPlan", "PlanStep"]

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    from bilt.core._evaluation_request import EvaluationRequest
    from bilt.core._model_constituent import BiltModelConstituent


@dataclasses.dataclass(frozen=True)
class PlanStep:
    """A single step in an analysis plan.

    Pairs a single model constituent with the output it is solving for and inputs needed for it.
    For transforms, ``output`` equals the transform's ``__bilt_name__``.
    For equations, ``output`` is whichever parameter is being isolated.

    For sub-systems, ``child_inputs`` contains the child-local ports the selected route requires,
    and is None on graph nodes before a route has been chosen. It is excluded from equality so a
    step handed back from realization still finds its node.

    This is intentionally decoupled from the actual isolation or system evaluation logic which
    takes place in _realize. We are only looking to determine if the path is valid through the
    system graph rather than if the steps can actually be performed.
    """

    operation: BiltModelConstituent
    output: str
    child_inputs: frozenset[str] | None = dataclasses.field(
        default=None,
        compare=False,
    )

    @property
    def consumed(self) -> tuple[str, ...]:
        """Return the attributes this step reads to produce its output, in a stable order."""
        match self.operation:
            case BasicTransform() as transform:
                return tuple(sorted(transform.__bilt_parameters__))

            case SubSystemConstituent() as sub_system:
                if self.child_inputs is None:
                    msg = f"{sub_system!r} step for {self.output!r} has not settled on a route through the child."
                    raise BiltAnalysisError(msg)

                return tuple(sorted(sub_system.qualify(port) for port in self.child_inputs))

            case BasicEquation() as equation:
                return tuple(sorted(set(equation.__bilt_parameters__) - {self.output}))

            case _:
                msg = f"Constituent type {type(self.operation)} is not supported."
                raise NotImplementedError(msg)


@dataclasses.dataclass(frozen=True)
class AnalysisPlan:
    """A plan through a complete system graph to satisfy an evaluation request."""

    request: EvaluationRequest
    steps: tuple[PlanStep, ...]

    @property
    def inputs(self) -> frozenset[str]:
        """Return the values available at the start of the plan."""
        return self.request.available

    @property
    def output(self) -> str:
        """Return the output the plan computes."""
        return self.request.output
