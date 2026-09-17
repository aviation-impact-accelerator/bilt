"""Errors for core system definitions, equations, analysis, and evaluation."""

__all__ = [
    "BiltAnalysisError",
    "BiltDefinitionError",
    "BiltDomainRegistryError",
    "BiltDomainRegistryLockedError",
    "BiltEquationReturnError",
    "BiltEquationSignatureError",
    "BiltEvaluationError",
    "BiltNameError",
    "BiltNoActiveEvaluationError",
]

from bilt._base_error import BiltError


class BiltDefinitionError(BiltError):
    """Raised when a system or equation definition is invalid."""


class BiltNameError(BiltDefinitionError):
    """Raised when a declared name collides with the separator reserved for flattened port names."""


class BiltEquationReturnError(BiltDefinitionError, TypeError):
    """Raised when an equation function does not return a Relation."""


class BiltEquationSignatureError(BiltDefinitionError, TypeError):
    """Raised when an equation function signature is invalid."""


class BiltAnalysisError(BiltError):
    """Raised when no evaluation route can satisfy a request."""


class BiltEvaluationError(BiltError):
    """Raised when evaluation cannot proceed."""


class BiltNoActiveEvaluationError(BiltEvaluationError, RuntimeError):
    """Raised when an operation requires an active evaluation context that is not available."""


class BiltDomainRegistryError(BiltEvaluationError, ValueError):
    """Raised when dimensions have conflicting evaluation bindings."""


class BiltDomainRegistryLockedError(BiltDomainRegistryError, RuntimeError):
    """Raised when trying to bind a new dimension to a locked domain registry."""
