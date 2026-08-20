__all__ = ["StructuralAnalyser"]

import typing

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt._backend._case import BackendCase


class StructuralAnalyser(typing.Protocol):
    """Protocol for a structural analyser that searches for a solution plan."""

    def analyse(self, available: set[str], output: str, concrete_backend: Backend) -> BackendCase: ...
