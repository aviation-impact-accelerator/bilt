import typing

import pytest

from bilt.core._structural_analysis import StructuralAnalyser
from bilt.core._structural_analysis._causalization_block_solver._backend import (
    CausalizationStructuralAnalyzer,
)
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._system import _implementation as implementation_module

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent


type AnalyserFactory = typing.Callable[
    [typing.Collection[BiltModelConstituent], Backend],
    StructuralAnalyser,
]


@pytest.fixture(
    params=(ExhaustiveStructuralAnalyser, CausalizationStructuralAnalyzer),
    ids=("exhaustive", "causalization"),
)
def analyser_factory(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> AnalyserFactory:
    factory = typing.cast("AnalyserFactory", request.param)

    monkeypatch.setattr(
        implementation_module,
        "ExhaustiveStructuralAnalyser",
        factory,
    )
    return factory
