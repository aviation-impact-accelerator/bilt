__all__ = [
    "BiltSystemImplementationRegistrationError",
    "BiltUnknownSystemAttributeError",
]

import typing

if typing.TYPE_CHECKING:
    from ._interface import Interface


class BiltSystemImplementationRegistrationError(Exception):
    pass


class BiltUnknownSystemAttributeError(Exception):
    name: str

    def __init__(self, name: str, interface: Interface) -> None:
        self.message = (
            f"Cannot evaluate attribute {name!r} because it was not set during initialization and no implementation "
            f"has been selected for this instance of the {interface!r} interface. Set an implementation for the "
            f"instance. "
        )
        super().__init__(self.message)
