import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bilt._backend import BackendCallable, BackendLinearSystem


@dataclasses.dataclass
class BackendCase:
    """Container for information expected by the backend to allow it to chain computations into a larger pipelined callable."""

    name: str
    parameters: tuple[str, ...]
    intermediates: Sequence[BackendCallable | BackendLinearSystem]
    return_name: str
