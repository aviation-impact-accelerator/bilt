"""Errors for core system definitions, equations, analysis, and evaluation."""

__all__ = [
    "BiltAnalysisError",
    "BiltDefinitionError",
    "BiltDomainBindingError",
    "BiltDomainRegistryError",
    "BiltDomainRegistryLockedError",
    "BiltEquationReturnError",
    "BiltEquationSignatureError",
    "BiltEvaluationError",
    "BiltNameError",
    "BiltNoActiveEvaluationError",
    "BiltPortDeclarationError",
    "BiltTransformDimensionalityError",
]

from bilt._base_error import BiltError
from bilt.units._errors import BiltDimensionalityError


class BiltDefinitionError(BiltError):
    """Raised when a system or equation definition is invalid."""


class BiltNameError(BiltDefinitionError):
    """Raised when a declared name collides with the separator reserved for flattened port names."""


class BiltEquationReturnError(BiltDefinitionError, TypeError):
    """Raised when an equation function does not return a Relation."""


class BiltEquationSignatureError(BiltDefinitionError, TypeError):
    """Raised when an equation function signature is invalid."""


class BiltPortDeclarationError(BiltDefinitionError, TypeError):
    """Raised when a port's annotation declares a dimensionality in a form bilt cannot read, or declares a unit."""

    def __init__(self, owner_name: str, port: str, problem: str) -> None:
        super().__init__(f"Port {port!r} of {owner_name!r} {problem}")


class BiltTransformDimensionalityError(BiltDimensionalityError):
    """Raised when probing a transform's body finds it inhomogeneous or contradicting its return annotation."""

    def __init__(self, transform_name: str, problem: str) -> None:
        super().__init__(f"Transform {transform_name!r} {problem}")


class BiltAnalysisError(BiltError):
    """Raised when no evaluation route can satisfy a request."""


class BiltEvaluationError(BiltError):
    """Raised when evaluation cannot proceed."""


class BiltNoActiveEvaluationError(BiltEvaluationError, RuntimeError):
    """Raised when an operation requires an active evaluation context that is not available."""


class BiltDomainRegistryError(BiltEvaluationError):
    """Base for failures binding or looking up dimensions in a domain registry."""


class BiltDomainBindingError(BiltDomainRegistryError, ValueError):
    """Raised when dimensions have conflicting evaluation bindings."""


class BiltDomainRegistryLockedError(BiltDomainRegistryError, RuntimeError):
    """Raised when trying to bind a new dimension to a locked domain registry."""
