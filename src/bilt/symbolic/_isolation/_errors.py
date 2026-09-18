"""Errors for symbolic equation isolation."""

__all__ = [
    "BiltIsolationError",
    "BiltTargetNotFoundError",
    "BiltTargetOnBothSidesError",
    "BiltUnsupportedIsolationError",
]

from bilt.symbolic._errors import BiltSymbolicError


class BiltIsolationError(BiltSymbolicError):
    """Base class for symbolic isolation failures."""


class BiltTargetNotFoundError(BiltIsolationError):
    """Raised when the target does not occur in the equation."""


class BiltTargetOnBothSidesError(BiltIsolationError):
    """Raised when the target occurs on both sides of the equation."""


class BiltUnsupportedIsolationError(BiltIsolationError):
    """Raised when the target occurs in a form the isolator cannot handle."""
