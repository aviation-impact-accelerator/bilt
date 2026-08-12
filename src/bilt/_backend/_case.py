import dataclasses
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from bilt._backend import BackendCallable
    from bilt.symbolic._domains import CanonicalisedDomain


@dataclasses.dataclass
class BackendCase:
    """Container for information expected by the backend to allow it to chain computations into a larger pipelined callable."""

    name: str
    parameters: dict[str, CanonicalisedDomain]
    intermediates: dict[str, BackendCallable]
    return_domain: CanonicalisedDomain
    return_name: str
