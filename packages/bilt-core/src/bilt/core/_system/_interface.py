__all__ = ["Interface"]

import typing

if typing.TYPE_CHECKING:
    from ._implementation import Implementation


class Interface(type):
    __bilt_system_registry__: set[Implementation]
    __bilt_system_implementation_default__: list[Implementation]

    def __new__(mcls: type[Interface], name: str, bases: tuple[type, ...], namespace: dict[str, object]) -> Interface:
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__bilt_system_registry__ = set()
        cls.__bilt_system_implementation_default__ = []
        return cls
