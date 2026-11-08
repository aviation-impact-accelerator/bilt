"""Small, unit-independent tensor foundation for the architecture experiment.

This is deliberately not Bilt's production Tensor. Its purpose is to prove
that the common quantity representation can be built without a tensor -> units
dependency. Production tensor behaviour is outside this experiment.
"""

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Tensor:
    """A numeric array with named axes, without a unit or quantity dependency."""

    values: np.ndarray
    domain: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != len(self.domain):
            message = "Tensor rank must match its domain"
            raise ValueError(message)
        if len(set(self.domain)) != len(self.domain):
            message = "Tensor domain cannot contain duplicate axes"
            raise ValueError(message)
        object.__setattr__(self, "values", values)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the concrete array shape."""
        return self.values.shape

    def sum(self, dim: str) -> float | Tensor:
        """Sum one named axis; return a scalar if no axes remain."""
        axis = self.domain.index(dim)
        remaining = tuple(name for name in self.domain if name != dim)
        reduced = np.sum(self.values, axis=axis)
        return Tensor(np.asarray(reduced), remaining) if remaining else float(reduced)


def tensor(values: object, *, over: tuple[str, ...]) -> Tensor:
    """Construct a unitless tensor over named axes."""
    return Tensor(np.asarray(values), over)
