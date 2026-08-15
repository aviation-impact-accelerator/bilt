import typing

if typing.TYPE_CHECKING:
    from bilt._backend import Backend, BackendCase
    from bilt.core._structural_analysis._plan import AnalysisPlan


@typing.runtime_checkable
class StructuralAnalyser(typing.Protocol):
    def analyse(
        self,
        available: set[str],
        output: str,
        concrete_backend: Backend,
    ) -> BackendCase: ...

    def plan_for_target(
        self,
        available: set[str],
        output: str,
    ) -> AnalysisPlan | None: ...

    def required_inputs_for_target(
        self,
        available: set[str],
        output: str,
    ) -> frozenset[str] | None: ...
