"""Errors for backend compilation and evaluation."""

__all__ = ["BiltBackendError", "BiltCompilationError", "BiltUnsupportedNodeError"]

from bilt._errors import BiltError


class BiltBackendError(BiltError):
    """Base error for backend compilation and evaluation failures."""


class BiltUnsupportedNodeError(BiltBackendError):
    """Raised when a backend does not support a particular operation node."""


class BiltCompilationError(BiltBackendError):
    """Raised when compilation to a backend fails."""
