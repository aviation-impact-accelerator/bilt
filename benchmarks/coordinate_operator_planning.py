"""Cold source-to-target coordinate operator planning trials."""

# ruff: noqa: INP001 - benchmark helper, not an installed package

import dataclasses
import typing
from contextlib import ExitStack

import numpy as np

from bilt import Implementation, Interface, Tensor, dim, domain, equation, field, system, tensor
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._system._lowering_and_lifting import lower_inputs
from bilt.core._tensors._tensor_alignment import generate_alignment_plan

if typing.TYPE_CHECKING:
    from bilt._backend import BackendCase

SOURCE = dim("source")
TARGET = dim("target")
Influence = typing.Annotated[Tensor, domain(TARGET, SOURCE)]
Circulation = typing.Annotated[Tensor, domain(SOURCE)]
Velocity = typing.Annotated[Tensor, domain(TARGET)]


@system
class SourceTarget(metaclass=Interface):
    """Simple inverse tensor operator used by the planning benchmarks."""

    influence: Influence = field()
    circulation: Circulation = field()
    velocity: Velocity = field()


@equation
def influence_equation(influence: Influence, circulation: Circulation, velocity: Velocity) -> bool:
    """Express the target velocity as the sum of source contributions."""
    return (influence * circulation).sum(SOURCE) == velocity


@dataclasses.dataclass
class CoordinateOperatorTrial:
    """One independent planning and execution trial with a known solution."""

    size: int
    implementation: Implementation
    values: dict[str, Tensor]
    request: EvaluationRequest
    resources: ExitStack
    expected: np.ndarray
    result: BackendCase | None = None
    checked: bool = False

    @classmethod
    def prepare(cls, size: int) -> typing.Self:
        """Create fresh coefficient inputs and domain bindings outside timing."""
        resources = ExitStack()
        try:
            a = 4.0 * np.eye(size) + 0.3 * np.ones((size, size))
            expected = np.linspace(0.5, 1.5, size)
            values = {
                "influence": tensor(a, over=(TARGET, SOURCE)),
                "velocity": tensor(a @ expected, over=TARGET),
            }
            context = resources.enter_context(EvaluationContext())
            context.domains.bind_all(values.values())
            implementation = Implementation(SourceTarget, [influence_equation])
            request = EvaluationRequest(
                output="circulation",
                input_domains={name: value.domain for name, value in values.items()},
            )
            return cls(size, implementation, values, request, resources, expected)
        except BaseException:
            resources.close()
            raise

    def analyse(self) -> None:
        """Compile a fresh inverse operator with existing structural analysis."""
        if self.result is not None:
            message = "A cold structural trial may only be analysed once."
            raise RuntimeError(message)
        self.result = self.implementation.analyser_for(PYTHON_BACKEND).analyse(self.request)

    def check(self) -> dict[str, object]:
        """Check computed circulation after the timed planning operation."""
        if self.result is None:
            message = "Coordinate operator trial returned no compiled plan."
            raise AssertionError(message)
        values: dict[str, object] = dict(self.values)
        parameters = PYTHON_BACKEND.apply_alignment(
            lower_inputs(values),
            generate_alignment_plan(values),
        )
        result = np.asarray(PYTHON_BACKEND.chain_case(self.result).callable(**parameters)).reshape(-1)
        if not np.allclose(result, self.expected, rtol=1e-8, atol=1e-8):
            message = f"Incorrect {self.size}-panel circulation: {result!r}"
            raise AssertionError(message)
        self.checked = True
        return {
            "status": "solved",
            "panels": self.size,
            "unknown_coordinates": self.size,
            "residual_coordinates": self.size,
            "max_error": float(np.max(np.abs(result - self.expected))),
            "finite": bool(np.isfinite(result).all()),
        }

    def close(self) -> None:
        """Release trial-scoped evaluation bindings."""
        self.resources.close()
