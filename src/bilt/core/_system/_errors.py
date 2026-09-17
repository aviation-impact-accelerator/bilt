"""Errors for system definitions and sub-system declarations."""

__all__ = [
    "BiltSubSystemConflictError",
    "BiltSubSystemError",
    "BiltSubSystemMismatchedImplementationError",
    "BiltSubSystemMissingImplementationError",
    "BiltSubSystemNameCollisionError",
    "BiltSubSystemUnreferencedEntryError",
    "BiltSystemDefinitionError",
    "BiltUnknownSystemAttributeError",
]

import typing

from bilt.core._errors import BiltDefinitionError, BiltEvaluationError

if typing.TYPE_CHECKING:
    from bilt.core._system._instance import SystemInstance


def _look_alike_hint(expected: str, actual: str) -> str:
    """Explain a message that names one interface twice, which otherwise reads as a bug in bilt.

    Interfaces are matched by identity, so two classes declared separately are two interfaces even
    when they carry the same name -- which is easy to do when each is declared inside a function.
    """
    if expected != actual:
        return ""
    return (
        " These are separate classes that happen to share a name: an interface is matched by "
        "identity, so an annotation and its implementation must name the very same class."
    )


class BiltSystemDefinitionError(BiltDefinitionError):
    """Raised when `@system` is applied to a class that does not declare `metaclass=Interface`."""

    cls_name: str

    def __init__(self, cls: type) -> None:
        self.cls_name = cls.__name__
        super().__init__(
            f"Cannot apply `@system` to {cls.__name__!r} because it does not declare `metaclass=Interface`. "
            f"Add `metaclass=Interface` to the class definition: `class {cls.__name__}(metaclass=Interface): ...`"
        )


class BiltUnknownSystemAttributeError(BiltEvaluationError):
    """Raised when an attribute was not set during initialization and no implementation has been selected."""

    name: str

    def __init__(self, name: str, instance: SystemInstance) -> None:
        self.name = name
        super().__init__(
            f"Cannot evaluate attribute {name!r} because it was not set during initialization and no implementation "
            f"has been selected for this instance of the {instance.__class__.__name__!r} interface. Set an "
            f"implementation for the instance."
        )


class BiltSubSystemError(BiltDefinitionError):
    """Base for every way a parent can declare its sub-systems wrongly.

    Each subclass names one condition so a caller can single it out, but they share a base so
    "this model's nesting does not add up" can be caught as one category.
    """

    reference_name: str


class BiltSubSystemMissingImplementationError(BiltSubSystemError):
    """Raised when an equation names a sub-system that nothing supplies an implementation for."""

    def __init__(self, reference_name: str, interface_name: str, provided_names: list[str]) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} (interface {interface_name!r}) has no entry in "
            f"internal_implementations. Provided names: {provided_names!r}"
        )


class BiltSubSystemMismatchedImplementationError(BiltSubSystemError):
    """Raised when the implementation supplied for a sub-system implements a different interface."""

    def __init__(self, reference_name: str, expected_interface_name: str, actual_interface_name: str) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} expects interface {expected_interface_name!r} "
            f"but the implementation supplied for it has interface {actual_interface_name!r}."
            + _look_alike_hint(expected_interface_name, actual_interface_name)
        )


class BiltSubSystemUnreferencedEntryError(BiltSubSystemError):
    """Raised when an ``internal_implementations`` entry answers to no equation."""

    def __init__(self, reference_name: str, referenced_names: list[str]) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"internal_implementations has entry {reference_name!r} but no equation references it. "
            f"Referenced sub-systems: {referenced_names!r}"
        )


class BiltSubSystemNameCollisionError(BiltSubSystemError):
    """Raised when a sub-system reference is spelled the same as one of the parent's own components."""

    def __init__(self, reference_name: str) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} collides with a component name. "
            f"A sub-system may not be named after any component's `__bilt_name__`."
        )


class BiltSubSystemConflictError(BiltSubSystemError):
    """Raised when two equations name one sub-system but annotate it with different interfaces."""

    def __init__(self, reference_name: str, declared_interface_name: str, other_interface_name: str) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} is used with different interfaces: "
            f"{[declared_interface_name, other_interface_name]!r}. "
            f"Every equation referencing a sub-system must annotate it with the same interface."
            + _look_alike_hint(declared_interface_name, other_interface_name)
        )
