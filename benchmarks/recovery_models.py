"""Failure exhaustion followed by a supported direction on the same analyser."""

# ruff: noqa: INP001 - benchmark helper, not an installed package

import dataclasses
import typing
from contextlib import ExitStack
from unittest.mock import patch

from bilt import Implementation, Interface, equation, field, system
from bilt._backend._case import BackendCase
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._system import _implementation as implementation_module
from bilt.errors import BiltIsolationError, BiltUnsupportedNodeError
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend._callable import BackendCallable


@system
class RecoveryPorts(metaclass=Interface):
    """Boundary values for a reversible relation and its wrappers."""

    x: float = typing.cast("float", field())
    z: float = typing.cast("float", field())


@equation
def square(x: float, z: float) -> bool:
    """Allow forward evaluation while inverse isolation is unsupported."""
    return z == x * x


@equation
def double(x: float, z: float) -> bool:
    """Provide the control and backend-refusal relation."""
    return z == 2 * x


@equation
def input_x(x: float, child: RecoveryPorts) -> bool:
    """Connect the parent's first port to the child."""
    return x == child.x


@equation
def output_z(z: float, child: RecoveryPorts) -> bool:
    """Connect the parent's second port to the child."""
    return z == child.z


def build_recovery_model(mode: str, depth: int) -> Implementation:
    """Wrap a single relation without introducing competing constraints."""
    if mode not in {"control", "isolation", "backend"} or depth < 0:
        raise ValueError((mode, depth))
    implementation = Implementation(RecoveryPorts, [square if mode == "isolation" else double])
    for _ in range(depth):
        implementation = Implementation(
            RecoveryPorts, [input_x, output_z], internal_implementations={"child": implementation}
        )
    return implementation


@dataclasses.dataclass
class RecoveryTrial:
    """A fresh analyser serving an inverse request and then a forward request."""

    solver: str
    mode: str
    implementation: Implementation
    resources: ExitStack
    refused: list[str]
    results: list[BackendCase | BiltAnalysisError] = dataclasses.field(default_factory=list)
    checked: bool = False
    closed: bool = False
    started: bool = False

    @classmethod
    def prepare(cls, solver: str, mode: str, depth: int) -> typing.Self:
        """Construct fixtures and install the optional scoped backend refusal."""
        if solver not in {"exhaustive", "scc"}:
            raise ValueError(solver)
        resources = ExitStack()
        try:
            factory = ExhaustiveStructuralAnalyser if solver == "exhaustive" else CausalizationStructuralAnalyzer
            resources.enter_context(patch.object(implementation_module, "CausalizationStructuralAnalyzer", factory))
            implementation = build_recovery_model(mode, depth)
            refused: list[str] = []
            if mode == "backend":
                original = PYTHON_BACKEND.transpile

                def refuse_inverse(function: ast.FunctionDef) -> BackendCallable:
                    if function.name == "x":
                        refused.append(function.name)
                        msg = "Benchmark backend refuses the inverse direction x."
                        raise BiltUnsupportedNodeError(msg)  # noqa: TRY301 - intentional backend callback failure
                    return original(function)

                resources.enter_context(patch.object(PYTHON_BACKEND, "transpile", refuse_inverse))
            resources.enter_context(EvaluationContext())
            return cls(solver, mode, implementation, resources, refused)
        except BaseException:
            resources.close()
            raise

    def analyse(self) -> None:
        """Time two requests on one initially fresh analyser, including recovery."""
        if self.closed or self.started:
            msg = "Each recovery trial must run once."
            raise RuntimeError(msg)
        self.started = True
        analyser = self.implementation.analyser_for(PYTHON_BACKEND)
        for available, output in (("z", "x"), ("x", "z")):
            request = EvaluationRequest(output=output, input_domains={available: Domain.scalar()})
            try:
                self.results.append(analyser.analyse(request))
            except BiltAnalysisError as error:
                self.results.append(error)

    def check(self) -> dict[str, object]:
        """Check the precise failure cause and the surviving direction's value."""
        expected_requests = 2
        if len(self.results) != expected_requests:
            msg = "Both recovery requests must complete."
            raise AssertionError(msg)
        first, second = self.results
        if self.mode == "control":
            inverse_expected = 5.0
            if (
                not isinstance(first, BackendCase)
                or PYTHON_BACKEND.chain_case(first).callable(z=10.0) != inverse_expected
            ):
                msg = f"Expected successful inverse control, received {first!r}."
                raise AssertionError(msg)
        else:
            expected_type = BiltIsolationError if self.mode == "isolation" else BiltUnsupportedNodeError
            cause: BaseException | None = first if isinstance(first, BiltAnalysisError) else None
            seen: set[int] = set()
            while isinstance(cause, BiltAnalysisError) and id(cause) not in seen:
                seen.add(id(cause))
                cause = cause.__cause__
            if not isinstance(cause, expected_type):
                msg = f"Expected {expected_type.__name__} as the rejection cause, received {first!r}."
                raise AssertionError(msg)
            if self.mode == "backend" and self.refused != ["x"]:
                msg = f"Expected one refused inverse compilation, received {self.refused!r}."
                raise AssertionError(msg)
        expected = 25.0 if self.mode == "isolation" else 10.0
        if not isinstance(second, BackendCase) or PYTHON_BACKEND.chain_case(second).callable(x=5.0) != expected:
            msg = f"Expected supported forward direction with z={expected}, received {second!r}."
            raise AssertionError(msg)
        self.checked = True
        return {
            "status": "success" if self.mode == "control" else "rejected_then_success",
            "expected_value": expected,
            "recovery_scope": "new_request_same_analyser",
            "refused_compilations": len(self.refused),
        }

    def close(self) -> None:
        """Release the evaluation context and restore solver/backend selection."""
        self.resources.close()
        self.closed = True
