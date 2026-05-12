__all__ = [
    "BiltUnknownSystemAttributeError",
]

import typing

from bilt.core._error import BiltCoreError

if typing.TYPE_CHECKING:
    from ._interface import Interface


class BiltUnknownSystemAttributeError(BiltCoreError):
    name: str

    def __init__(self, name: str, interface: Interface) -> None:
        self.message = (
            f"Cannot evaluate attribute {name!r} because it was not set during initialization and no implementation "
            f"has been selected for this instance of the {interface.__class__!r} interface. Set an implementation for "
            f"the instance."
        )
        super().__init__(self.message)
