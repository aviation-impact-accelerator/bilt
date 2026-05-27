__all__ = []

import typing

if typing.TYPE_CHECKING:
    from ._implementation import Implementation
    from ._sentinel import _Missing


class SystemInstance(typing.Protocol):
    __bilt_system_instance_variables__: dict[str, object]
    __bilt_system_implementation__: Implementation | _Missing
