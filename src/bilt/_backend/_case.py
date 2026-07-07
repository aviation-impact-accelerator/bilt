import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilt._backend import BackendCallable


@dataclasses.dataclass
class BackendCase:
    """Container for information expected by the backend to allow it to chain computations into a larger pipelined callable."""

    name: str
    parameters: tuple[str, ...]
    intermediates: list[BackendCallable]
    return_name: str
