"""Errors for backend compilation and evaluation."""

__all__ = [
    "BiltBackendError",
    "BiltCompilationError",
    "BiltLinearSolveError",
    "BiltNonlinearSolveError",
    "BiltUnsupportedNodeError",
]

from bilt._base_error import BiltError


class BiltBackendError(BiltError):
    """Base error for backend compilation and evaluation failures."""


class BiltUnsupportedNodeError(BiltBackendError):
    """Raised when a backend does not support a particular operation node."""


class BiltCompilationError(BiltBackendError):
    """Raised when compilation to a backend fails."""


class BiltLinearSolveError(BiltBackendError):
    """A linear system could not be evaluated safely."""

    def __init__(self, problem: str) -> None:
        self.problem = problem
        super().__init__(f"Cannot solve linear block: {problem}.")


class BiltNonlinearSolveError(BiltBackendError):
    """A scalar nonlinear solve failed without invalidating its prepared structure."""

    def __init__(self, problem: str, *, evaluations: int = 0, residual_norm: float | None = None) -> None:
        self.problem = problem
        self.evaluations = evaluations
        self.residual_norm = residual_norm
        super().__init__(f"Cannot solve nonlinear block: {problem}.")
