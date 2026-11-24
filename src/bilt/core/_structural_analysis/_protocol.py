__all__ = ["StructuralAnalyser"]

import typing

if typing.TYPE_CHECKING:
    from bilt._backend._case import BackendCase
    from bilt.core._evaluation_request import EvaluationRequest


class StructuralAnalyser(typing.Protocol):
    """Protocol for a structural analyser that searches for a solution plan for a certain backend."""

    def analyse(self, request: EvaluationRequest) -> BackendCase: ...

    def required_inputs_for_target(self, available: set[str], output: str) -> frozenset[str] | None: ...
