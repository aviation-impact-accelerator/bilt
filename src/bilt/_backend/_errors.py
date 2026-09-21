"""Errors for backend compilation and evaluation."""

__all__ = [
    "BiltBackendError",
    "BiltBackendOutputArityError",
    "BiltCompilationError",
    "BiltUnsupportedNodeError",
]

from bilt._base_error import BiltError


class BiltBackendError(BiltError):
    """Base error for backend compilation and evaluation failures."""


class BiltUnsupportedNodeError(BiltBackendError):
    """Raised when a backend does not support a particular operation node."""


class BiltCompilationError(BiltBackendError):
    """Raised when compilation to a backend fails."""


class BiltBackendOutputArityError(BiltBackendError, ValueError):
    """Raised when a backend callable returns a different number of values than it names."""
