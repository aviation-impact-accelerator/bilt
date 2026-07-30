from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend
    from bilt._backend._case import BackendCase
    from bilt.core._model_constituent import BiltModelConstituent


class StructuralAnalysisBackend(typing.Protocol):
    def __call__(
        self,
        constituents: typing.Collection[BiltModelConstituent],
        available: set[str],
        output: str,
        concrete_backend: Backend,
    ) -> BackendCase: ...
