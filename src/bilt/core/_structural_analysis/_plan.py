__all__ = ["AnalysisPlan", "PlanStep"]

import dataclasses
import typing

from bilt.core._equation import BasicEquation
from bilt.core._error import BiltSystemAnalysisError
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

    For sub-systems, ``request`` is the child-local evaluation the search settled on, and is None
    on graph nodes, which stand for a sub-system before any direction is chosen. It is excluded
    from equality so a step handed back from realization still finds its node.

    This is intentionally decoupled from the actual isolation or system evaluation logic which takes place in _realize.
    We are only looking to determine if the path is valid through the system graph rather than if the steps can actually be performed.
    """

    operation: BiltModelConstituent
    output: str
    request: EvaluationRequest | None = dataclasses.field(default=None, compare=False)

    @property
    def consumed(self) -> tuple[str, ...]:
        """Return the attributes this step reads to produce its output, in a stable order."""
        match self.operation:
            case BasicTransform() as transform:
                return tuple(sorted(transform.__bilt_parameters__))
            case SubSystemConstituent() as sub_system:
                if self.request is None:
                    msg = f"{sub_system!r} step for {self.output!r} has not settled on a route through the child."
                    raise BiltSystemAnalysisError(msg)
                return tuple(sorted(sub_system.qualify(port) for port in self.request.available))
            case BasicEquation() as equation:
                return tuple(sorted(set(equation.__bilt_parameters__) - {self.output}))
            case _:
                msg = f"Constituent type {type(self.operation)} is not supported."
                raise NotImplementedError(msg)


@dataclasses.dataclass(frozen=True)
class AnalysisPlan:
    """A plan through a complete system graph to compute a desired output from available inputs."""

    inputs: frozenset[str]
    steps: tuple[PlanStep, ...]
    output: str
