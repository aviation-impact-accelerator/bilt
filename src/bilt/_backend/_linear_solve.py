import dataclasses
import typing

if typing.TYPE_CHECKING:
    from bilt._backend._callable import BackendCallable


@dataclasses.dataclass(frozen=True, slots=True)
class BackendLinearSystem:
    """A concrete linear system of the form A x = b."""

    parameters: tuple[str, ...]
    unknowns: tuple[str, ...]
    coefficients: tuple[tuple[BackendCallable, ...], ...]
    rhs: tuple[BackendCallable, ...]
