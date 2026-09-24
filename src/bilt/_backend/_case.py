import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import collections.abc

    from bilt._backend import BackendCallable, BackendLinearSystem


@dataclasses.dataclass
class BackendCase:
    """A backend-ready sequence of executable operations."""

    name: str
    parameters: tuple[str, ...]
    intermediates: collections.abc.Sequence[BackendCallable | BackendLinearSystem]
    return_name: str
