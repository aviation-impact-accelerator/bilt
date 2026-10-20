__all__ = ["CausalizationStructuralAnalyzer"]

import typing

from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph

if typing.TYPE_CHECKING:
    from bilt._backend import Backend
    from bilt.core._model_constituent import BiltModelConstituent


class CausalizationStructuralAnalyzer:
    """Analyser that uses matching and SCC decomposition to construct a plan."""

    def __init__(self, constituents: typing.Collection[BiltModelConstituent], backend: Backend) -> None:
        self.backend = backend
        self.full_incidence_graph = FullIncidenceGraph(constituents)
