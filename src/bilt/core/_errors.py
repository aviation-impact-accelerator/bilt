"""Errors for core system definitions, equations, analysis, and evaluation."""

__all__ = [
    "BiltAnalysisError",
    "BiltDefinitionError",
    "BiltDomainBindingError",
    "BiltDomainRegistryError",
    "BiltDomainRegistryLockedError",
    "BiltEquationDimensionalityError",
    "BiltEquationReturnError",
    "BiltEquationSignatureError",
    "BiltEvaluationError",
    "BiltImplementationDimensionalityError",
    "BiltInputDimensionalityError",
    "BiltNameError",
    "BiltNoActiveEvaluationError",
    "BiltPortDeclarationError",
    "BiltUnresolvedAnnotationError",
]

import typing

from bilt._base_error import BiltError
from bilt.units import Quantity
from bilt.units._errors import BiltDimensionalityError

if typing.TYPE_CHECKING:
    from bilt.units import Dimension


class BiltDefinitionError(BiltError):
    """Raised when a system or equation definition is invalid."""


class BiltNameError(BiltDefinitionError):
    """Raised when a declared name collides with the separator reserved for flattened port names."""


class BiltEquationReturnError(BiltDefinitionError, TypeError):
    """Raised when an equation function does not return a Relation."""


class BiltEquationSignatureError(BiltDefinitionError, TypeError):
    """Raised when an equation function signature is invalid."""


class BiltPortDeclarationError(BiltDefinitionError, TypeError):
    """Raised when a port's annotation is not a readable declaration, or a constituent mixes declared and plain ports."""

    def __init__(self, owner_name: str, port: str | None, problem: str) -> None:
        subject = f"Port {port!r} of {owner_name!r}" if port is not None else f"Ports of {owner_name!r}"
        super().__init__(f"{subject} {problem}")


class BiltUnresolvedAnnotationError(BiltDefinitionError, NameError):
    """Raised when a decorated definition's annotations name something that is not defined at runtime."""

    owner_name: str

    def __init__(self, owner_name: str, error: NameError) -> None:
        self.owner_name = owner_name
        super().__init__(
            f"Cannot read the annotations of {owner_name!r}: {error}. bilt reads annotations when the decorator "
            f"runs, so every name they use must be imported or defined before {owner_name!r}, outside any "
            "`if typing.TYPE_CHECKING:` block.",
            name=error.name,
        )


class BiltEquationDimensionalityError(BiltDimensionalityError):
    """Raised when an equation's relation cannot be dimensionally homogeneous."""

    def __init__(self, equation_name: str, problem: str) -> None:
        super().__init__(f"Equation {equation_name!r} {problem}")


class BiltImplementationDimensionalityError(BiltDimensionalityError):
    """Raised when an implementation gives a port conflicting dimensionalities, or makes an equation inconsistent."""

    def __init__(self, system_name: str, problem: str) -> None:
        super().__init__(f"Implementation of {system_name!r} {problem}")


class BiltInputDimensionalityError(BiltDimensionalityError):
    """Raised when a value given for a port is not the quantity the port declares, or is a quantity for a plain port."""

    def __init__(self, system_name: str, port: str, expected: Dimension | None, value: object) -> None:
        received = (
            f"{value} of {value.unit.dimension}" if isinstance(value, Quantity) else f"{type(value).__name__} {value!r}"
        )
        if expected is None:
            msg = (
                f"Input {port!r} of {system_name!r} is a plain port, outside the unit system, "
                f"so it rejects a Quantity: got {received}."
            )
        else:
            msg = f"Input {port!r} of {system_name!r} must be a Quantity of {expected}, not {received}."
        super().__init__(msg)


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
