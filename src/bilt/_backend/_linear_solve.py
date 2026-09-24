# src/bilt/_backend/_linear_solve.py

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

    def __post_init__(self) -> None:
        size = len(self.unknowns)

        if size == 0:
            msg = "A linear system must contain at least one unknown."
            raise ValueError(msg)

        if len(self.coefficients) != size:
            msg = "Linear system coefficient matrix must be square."
            raise ValueError(msg)

        if any(len(row) != size for row in self.coefficients):
            msg = "Linear system coefficient matrix must be square."
            raise ValueError(msg)

        if len(self.rhs) != size:
            msg = "Linear system RHS must match the number of unknowns."
            raise ValueError(msg)