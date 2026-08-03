import typing

from bilt._backend._python._backend import PythonBackend
from bilt.core import transform
from bilt.core._structural_analysis import StructuralAnalysisBackend
from bilt.core._structural_analysis._exhaustive import exhaustive_structural_backend


def test_exhaustive_backend_is_a_structural_analysis_backend() -> None:
    # pyrefly: ignore[isinstance-against-protocol]
    assert isinstance(  # ty: ignore[isinstance-against-protocol]
        exhaustive_structural_backend,
        typing.runtime_checkable(StructuralAnalysisBackend),  # pyrefly: ignore[invalid-argument]
    )


def test_exhaustive_backend_can_be_called() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    backend = PythonBackend()
    result = exhaustive_structural_backend([c], {"a", "b"}, "c", backend)
    backend_case = backend.chain_case(result)
    assert backend_case.callable(a=1.0, b=2.0) == c(a=1.0, b=2.0)
