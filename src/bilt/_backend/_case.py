import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass
class BackendCase:
    """Container for information expected by the backend to allow it to chain computations into a larger pipelined callable."""

    name: str
    parameters: tuple[str, ...]
    interim: list[collections.abc.Callable]
    return_name: str
