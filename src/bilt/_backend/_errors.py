"""Errors for backend compilation and evaluation."""

__all__ = ["BiltBackendError", "BiltCompilationError", "BiltLinearSolveError", "BiltUnsupportedNodeError"]

from bilt._base_error import BiltError


class BiltBackendError(BiltError):
    """Base error for backend compilation and evaluation failures."""


class BiltUnsupportedNodeError(BiltBackendError):
    """Raised when a backend does not support a particular operation node."""


class BiltCompilationError(BiltBackendError):
    """Raised when compilation to a backend fails."""


class BiltLinearSolveError(BiltBackendError):
    """A scalar linear system could not be evaluated safely."""

    def __init__(self, problem: str) -> None:
        self.problem = problem
        super().__init__(f"Cannot solve scalar linear block: {problem}.")
