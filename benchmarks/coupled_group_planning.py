"""Cold planning benchmarks for linear groups larger than equation pairs."""

# ruff: noqa: INP001 - standalone benchmark companion

import dataclasses
import math
import typing
from contextlib import ExitStack

from coupled_group_models import build_group

from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt import Implementation
    from bilt._backend import BackendCase


@dataclasses.dataclass
class CoupledGroupTrial:
    """One independent structural planning and execution trial."""

    size: int
    implementation: Implementation
    inputs: dict[str, float]
    request: EvaluationRequest
    resources: ExitStack
    result: BackendCase | None = None
    checked: bool = False

    @classmethod
    def prepare(cls, size: int) -> typing.Self:
        """Build a fresh model and evaluation context outside timing."""
        resources = ExitStack()
        try:
            implementation, inputs = build_group(size)
            request = EvaluationRequest(output="x", input_domains=dict.fromkeys(inputs, Domain.scalar()))
            resources.enter_context(EvaluationContext())
            return cls(size, implementation, inputs, request, resources)
        except BaseException:
            resources.close()
            raise

    def analyse(self) -> None:
        """Build and realise the plan; do not reuse it across rounds."""
        if self.result is not None:
            message = "Already analysed"
            raise RuntimeError(message)
        self.result = self.implementation.analyser_for(PYTHON_BACKEND).analyse(self.request)

    def check(self) -> dict[str, object]:
        """Check the independent expected result after every timed round."""
        if self.result is None:
            message = "Planning did not return a result"
            raise AssertionError(message)
        value = PYTHON_BACKEND.chain_case(self.result).callable(**self.inputs)
        if not math.isclose(float(value), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            message = f"Expected x=1, got {value!r}"
            raise AssertionError(message)
        self.checked = True
        return {"status": "solved", "size": self.size, "x": float(value)}

    def close(self) -> None:
        """Release the current trial's evaluation context."""
        self.resources.close()
