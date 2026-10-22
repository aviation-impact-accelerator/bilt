__all__ = ["AnalysisPlan", "PlanBlock", "PlanStep"]

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
class PlanBlock:
    """An SCC of matched operations, with its external data dependencies.

    Outputs are the block's unknowns. Each step retains its original operation
    and selected child route. Step order is reproducible, but is not an
    execution order inside a cyclic block. A self-loop is cyclic even when the
    block contains only one step.
    """

    steps: tuple[PlanStep, ...]
    is_cyclic: bool = False

    def __post_init__(self) -> None:
        if not self.steps:
            msg = "A plan block must contain at least one step."
            raise ValueError(msg)
        if len(self.steps) > 1 and not self.is_cyclic:
            msg = "An acyclic SCC block must contain exactly one step."
            raise ValueError(msg)

    @property
    def outputs(self) -> frozenset[str]:
        """Return the unknowns produced together by this block."""
        return frozenset(step.output for step in self.steps)

    @property
    def inputs(self) -> frozenset[str]:
        """Return dependencies supplied from outside this block."""
        outputs = self.outputs
        return frozenset(attribute for step in self.steps for attribute in step.consumed if attribute not in outputs)

    @property
    def operations(self) -> tuple[BiltModelConstituent, ...]:
        """Return the matched operations in the same order as the steps."""
        return tuple(step.operation for step in self.steps)


@dataclasses.dataclass(frozen=True)
class AnalysisPlan:
    """Dependency-ordered blocks satisfying an evaluation request."""

    request: EvaluationRequest
    blocks: tuple[PlanBlock, ...]

    @property
    def steps(self) -> tuple[PlanStep, ...]:
        """Return a flat inspection view, not an execution order for cyclic blocks."""
        return tuple(step for block in self.blocks for step in block.steps)

    @property
    def inputs(self) -> frozenset[str]:
        """Return the values available at the start of the plan."""
        return frozenset(self.request.input_domains)

    @property
    def output(self) -> str:
        """Return the output the plan computes."""
        return self.request.output
