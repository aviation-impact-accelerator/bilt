__all__ = [
    "IsolationError",
    "TargetNotFoundError",
    "TargetOnBothSidesError",
    "UnsupportedIsolationError",
]


class IsolationError(Exception):
    """Base class for symbolic isolation failures."""


class TargetNotFoundError(IsolationError):
    """Raised when the target does not occur in the equation."""


class TargetOnBothSidesError(IsolationError):
    """Raised when the target occurs on both sides of the equation."""


class UnsupportedIsolationError(IsolationError):
    """Raised when the target occurs in a form the isolator cannot handle."""
