import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilt._backend import BackendCallable
    from bilt._backend._callable import BackendOutput


@dataclasses.dataclass
class BackendCase:
    """Pipeline that optionally returns named intermediate ports as a positional tuple."""

    name: str
    parameters: tuple[str, ...]
    intermediates: list[BackendCallable]
    return_name: str
    outputs: tuple[BackendOutput, ...] = ()
    expose_intermediates: bool = False
