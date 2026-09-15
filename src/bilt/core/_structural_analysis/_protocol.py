__all__ = ["StructuralAnalyser"]

import typing

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase


class StructuralAnalyser(typing.Protocol):
    """Protocol for a structural analyser that plans solutions for one backend."""

    def analyse(self, available: set[str], output: str) -> BackendCase: ...

    def required_inputs_for_target(self, available: set[str], output: str) -> frozenset[str] | None: ...
