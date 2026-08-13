import dataclasses
import typing

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCallable:
    name: str
    parameters: tuple[str, ...]
    callable: collections.abc.Callable[..., typing.Any]
