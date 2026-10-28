import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilt._backend import BackendCallable
    from bilt._backend._callable import BackendOutput


@dataclasses.dataclass
class BackendCase:
    """A pipeline with an optional positional return for its computed ports.

    ``outputs`` preserves the logical names and domains collected during realization.
    When ``expose_intermediates`` is true, return these values as an ordered tuple;
    otherwise return only ``return_name``, as required for subsystem composition.
    """

    name: str
    parameters: tuple[str, ...]
    intermediates: list[BackendCallable]
    return_name: str
    outputs: tuple[BackendOutput, ...] = ()
    expose_intermediates: bool = False
