__all__ = ["BiltSystemDefinitionError", "BiltUnknownSystemAttributeError"]

import typing

from bilt.core._error import BiltCoreError

if typing.TYPE_CHECKING:
    from ._interface import Interface


class BiltSystemDefinitionError(BiltCoreError):
    cls_name: str

    def __init__(self, cls: type) -> None:
        self.cls_name = cls.__name__
        self.message = (
            f"Cannot apply `@system` to {cls.__name__!r} because it does not declare `metaclass=Interface`. "
            f"Add `metaclass=Interface` to the class definition: `class {cls.__name__}(metaclass=Interface): ...`"
        )
        super().__init__(self.message)


class BiltUnknownSystemAttributeError(BiltCoreError):
    name: str

    def __init__(self, name: str, interface: Interface) -> None:
        self.name = name
        self.message = (
            f"Cannot evaluate attribute {name!r} because it was not set during initialization and no implementation "
            f"has been selected for this instance of the {interface.__class__.__name__!r} interface. Set an "
            f"implementation for the instance."
        )
        super().__init__(self.message)
