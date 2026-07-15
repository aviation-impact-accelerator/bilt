__all__ = ["Interface"]

import typing

if typing.TYPE_CHECKING:
    from ._decorator import Attribute
    from ._implementation import Implementation
    from ._sentinel import _Missing


@typing.dataclass_transform(kw_only_default=True)
class Interface(type):
    __bilt_name__: str
    __bilt_parameters__: tuple[str, ...]
    __bilt_system_registry__: set[Implementation]
    __bilt_system_implementation_default__: list[Implementation]

    __bilt_system_attributes__: typing.ClassVar[dict[str, Attribute[object]]]
    __bilt_system_implementation__: Implementation | _Missing
    __bilt_system_instance_variables__: dict[str, object]

    def __new__(mcls: type[Interface], name: str, bases: tuple[type, ...], namespace: dict[str, object]) -> Interface:
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__bilt_name__ = name

        cls.__bilt_system_registry__ = set()
        cls.__bilt_system_implementation_default__ = []
        return cls
