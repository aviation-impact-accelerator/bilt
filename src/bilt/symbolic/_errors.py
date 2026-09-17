"""Errors for symbolic computation and domain validation."""

__all__ = [
    "BiltAtomMutationError",
    "BiltInvalidDomainError",
    "BiltSymbolicConversionError",
    "BiltSymbolicError",
    "BiltTreeMutationError",
    "BiltUnknownPortError",
]

from bilt._errors import BiltError


class BiltSymbolicError(BiltError):
    """Base error for symbolic computation failures."""


class BiltAtomMutationError(BiltSymbolicError, AttributeError):
    """Raised when attempting to mutate an immutable Atom."""


class BiltTreeMutationError(BiltSymbolicError, AttributeError):
    """Raised when attempting to mutate an immutable Tree."""


class BiltUnknownPortError(BiltSymbolicError, AttributeError):
    """Raised when attempting to access a port that does not exist."""


class BiltSymbolicConversionError(BiltSymbolicError, TypeError):
    """Raised when a symbolic expression cannot be converted to the requested form."""


class BiltInvalidDomainError(BiltSymbolicError, ValueError):
    """Raised when a domain specification is invalid."""
