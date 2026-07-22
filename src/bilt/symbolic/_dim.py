__all__ = ["Dim", "dim"]

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class Dim:
    """A named symbolic axis."""

    name: str

    def __str__(self) -> str:
        return self.name


def dim(name: str) -> Dim:
    return Dim(name)
