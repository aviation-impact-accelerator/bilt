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
    from bilt.core._port_declaration import PortDeclaration


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
    """Raised when an implementation gives a port conflicting declarations, or makes an equation inconsistent."""

    def __init__(self, system_name: str, problem: str) -> None:
        super().__init__(f"Implementation of {system_name!r} {problem}")


class BiltInputDimensionalityError(BiltDimensionalityError):
    """Raised when a value given for a port is not the quantity the port declares."""

    def __init__(self, system_name: str, port: str, expected: PortDeclaration | None, value: object) -> None:
        # Deferred: `core._tensors` reaches this module for its own error's base class, so it cannot be imported here.
        from bilt.core._tensors._tensors import Tensor  # noqa: PLC0415

        label = ""
        match value:
            case Quantity(unit=unit):
                received = f"{value} of {unit.dimension}"
            case Tensor(unit=unit) if unit is None:
                received = f"a Tensor with no unit over {value.domain}"
                label = " Label it with `tensor(..., unit=...)`."
            case Tensor(unit=unit):
                received = f"a Tensor in {unit} over {value.domain}"
            case _:
                unit = None
                received = f"{type(value).__name__} {value!r}"

        carrier = "a Tensor" if expected is not None and expected.over_a_domain else "a Quantity"
        if expected is None:
            msg = (
                f"Input {port!r} of {system_name!r} is a plain port, outside the unit system, so it rejects a "
                f"{'unit-carrying Tensor' if isinstance(value, Tensor) else 'Quantity'}: got {received}."
            )
        elif expected.over_a_domain and not isinstance(value, Tensor):
            msg = (
                f"Input {port!r} of {system_name!r} varies over a domain, so it must be a Tensor of {expected}, "
                f"not {received}. Annotate it `Quantity` instead if it does not vary over a domain."
            )
        elif unit is not None and unit.dimension == expected.dimensionality:
            msg = (
                f"Input {port!r} of {system_name!r} must be {carrier} of {expected} in "
                f"{'an absolute' if expected.absolute else 'a relative'} unit such as {expected.coherent_unit}, "
                f"not {received}."
            )
        else:
            msg = f"Input {port!r} of {system_name!r} must be {carrier} of {expected}, not {received}.{label}"
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
