__all__ = [
    "BiltSubSystemConflictError",
    "BiltSubSystemMismatchedImplementationError",
    "BiltSubSystemMissingImplementationError",
    "BiltSubSystemNameCollisionError",
    "BiltSubSystemUnreferencedEntryError",
    "BiltSystemDefinitionError",
    "BiltUnknownSystemAttributeError",
]

import typing

from bilt.core._error import BiltCoreError

if typing.TYPE_CHECKING:
    from ._instance import SystemInstance


class BiltSystemDefinitionError(BiltCoreError):
    cls_name: str

    def __init__(self, cls: type) -> None:
        self.cls_name = cls.__name__
        super().__init__(
            f"Cannot apply `@system` to {cls.__name__!r} because it does not declare `metaclass=Interface`. "
            f"Add `metaclass=Interface` to the class definition: `class {cls.__name__}(metaclass=Interface): ...`"
        )


class BiltUnknownSystemAttributeError(BiltCoreError):
    name: str

    def __init__(self, name: str, instance: SystemInstance) -> None:
        self.name = name
        super().__init__(
            f"Cannot evaluate attribute {name!r} because it was not set during initialization and no implementation "
            f"has been selected for this instance of the {instance.__class__.__name__!r} interface. Set an "
            f"implementation for the instance."
        )


class BiltSubSystemMissingImplementationError(BiltCoreError):
    reference_name: str

    def __init__(self, reference_name: str, interface_name: str, provided_keys: list[str]) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} (interface {interface_name!r}) has no entry in "
            f"internal_implementations. Provided keys: {provided_keys!r}"
        )


class BiltSubSystemMismatchedImplementationError(BiltCoreError):
    reference_name: str

    def __init__(self, reference_name: str, expected_interface_name: str, actual_interface_name: str) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} expects interface {expected_interface_name!r} "
            f"but internal_implementations entry has interface {actual_interface_name!r}"
        )


class BiltSubSystemUnreferencedEntryError(BiltCoreError):
    entry_name: str

    def __init__(self, entry_name: str, referenced_names: list[str]) -> None:
        self.entry_name = entry_name
        super().__init__(
            f"internal_implementations has entry {entry_name!r} but no equation references it. "
            f"Referenced sub-systems: {referenced_names!r}"
        )


class BiltSubSystemNameCollisionError(BiltCoreError):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"Sub-system reference {name!r} collides with a component name. "
            f"Sub-system names must not equal any component's __bilt_name__"
        )


class BiltSubSystemConflictError(BiltCoreError):
    reference_name: str

    def __init__(self, reference_name: str, interfaces: list[str]) -> None:
        self.reference_name = reference_name
        super().__init__(
            f"Sub-system reference {reference_name!r} is used with different interfaces: {interfaces!r}. "
            f"All equations referencing a sub-system must use the same interface."
        )
