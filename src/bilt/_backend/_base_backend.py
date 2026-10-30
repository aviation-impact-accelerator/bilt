"""Minimal interface for bilt backend, that will be utilised by the analysis layer."""

import typing

if typing.TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from bilt._backend._nonlinear_solve import NonlinearControls, NonlinearSolver
    from bilt.core._tensors._tensor_alignment import AlignmentPlan

    from ._callable import BackendCallable
    from ._case import BackendCase

_backend_stack: list[Backend] = []


class Backend(typing.Protocol):
    """Interface a backend must expose. `ActivatableBackendMixin` supplies the context manager half."""

    def transpile(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a single transform AST into a callable function, that can then be chained into the execution pipeline."""
        ...

    def chain_case(self, case: BackendCase) -> BackendCallable:
        """Chain the callables of a case into a single callable."""
        ...

    def __enter__(self) -> typing.Self: ...

    def __exit__(self, *_: object) -> None: ...

    def apply_alignment(self, lowered_inputs: dict[str, object], alignment_plan: AlignmentPlan) -> dict[str, object]:
        """Apply the given alignment plan on the provided lowered inputs."""


@typing.runtime_checkable
class LinearSolveBackend(typing.Protocol):
    """Optional compilation capability for linear-solve functions."""

    def transpile_linear_solve(self, function_definition: ast.FunctionDef) -> BackendCallable:
        """Compile a function containing Bilt's linear-solve intrinsic."""
        ...


class ActivatableBackendMixin:
    """Makes a backend the one dispatch selects for the duration of a `with` block."""

    def __enter__[T: Backend](self: T) -> T:  # noqa: PYI019
        _backend_stack.append(self)
        return self

    def __exit__(self, *_: object) -> None:
        _backend_stack.pop()


@typing.runtime_checkable
class NonlinearSolveBackend(typing.Protocol):
    """Optional direct execution capability for prepared scalar residuals."""

    def solve_nonlinear(  # noqa: PLR0913 - explicit solver, scales and numerical controls
        self,
        residual: Callable[[tuple[float, ...]], object],
        initial_guess: tuple[float, ...],
        *,
        variable_scales: tuple[float, ...],
        residual_scales: tuple[float, ...],
        controls: NonlinearControls,
        solver: NonlinearSolver | None = None,
    ) -> tuple[float, ...]: ...
