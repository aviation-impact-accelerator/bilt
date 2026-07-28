from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._plan import AnalysisPlan


class StructuralAnalysisBackend(typing.Protocol):
    def __call__(
        self,
        components: tuple[BiltModelConstituent, ...],
        inputs: frozenset[str],
        output: str,
    ) -> AnalysisPlan: ...
