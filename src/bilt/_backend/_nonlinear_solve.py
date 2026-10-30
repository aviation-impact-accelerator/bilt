"""Backend-independent controls and contracts for nonlinear solver adapters."""

import dataclasses
import math
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class NonlinearControls:
    """Controls in scaled coordinates; the budget includes validation evaluations."""

    max_evaluations: int = 200
    step_tolerance: float = 1e-8
    residual_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if type(self.max_evaluations) is not int or self.max_evaluations <= 0:
            message = "max_evaluations must be a positive integer."
            raise ValueError(message)
        for name in ("step_tolerance", "residual_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                message = f"{name} must be finite and positive."
                raise ValueError(message)


@dataclasses.dataclass(frozen=True)
class NonlinearResult[Vector]:
    """One algorithm's candidate and termination report, before Bilt validation.

    ``converged`` is the algorithm's own report, not permission to accept the
    candidate. The execution backend independently validates shape, finiteness
    and scaled residuals. Vectors retain the backend's native representation.
    """

    solution: Vector
    converged: bool
    message: str


@typing.runtime_checkable
class NonlinearSolver[Vector](typing.Protocol):
    """An interchangeable numerical algorithm for a specific execution backend.

    Adapters receive dimensionless scaled coordinates and residuals. Every
    residual evaluation, including numerical Jacobian evaluations, must use
    the supplied callable so the backend can enforce one evaluation budget.
    Return a NonlinearResult for normal convergence or non-convergence; raise
    BiltBackendError for missing dependencies or unsupported configuration.

    Algorithm-specific options belong on the adapter, not in core preparation.
    No registry, inheritance or changes to the planner are needed to add one.
    """

    @property
    def name(self) -> str:
        """A stable algorithm identifier for diagnostics."""
        ...

    @property
    def backend(self) -> str:
        """The execution backend whose native vector representation is expected."""
        ...

    def solve(
        self,
        residual: Callable[[Vector], Vector],
        initial_guess: Vector,
        *,
        controls: NonlinearControls,
    ) -> NonlinearResult[Vector]: ...
