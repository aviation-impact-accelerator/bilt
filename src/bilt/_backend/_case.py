import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bilt._backend import BackendCallable
    from bilt._backend._callable import DomainedVar


@dataclasses.dataclass
class BackendCase:
    """Container for information expected by the backend to allow it to chain computations into a larger pipelined callable."""

    name: str
    parameters: dict[str, DomainedVar]
    intermediates: dict[str, BackendCallable]
    returns: DomainedVar
