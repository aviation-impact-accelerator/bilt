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
    expose_intermediates: bool = True

    @property
    def output_names(self) -> tuple[str, ...]:
        """The values the case computes, named in the order the intermediates produce them."""
        return tuple(intermediate.name for intermediate in self.intermediates)
