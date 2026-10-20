__all__ = ["FullIncidenceGraph"]

import typing

import rustworkx as rx

from bilt.core._equation import BasicEquation
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent


class FullIncidenceGraph:
    """Undirected bipartite graph connecting constituents to their attributes."""

    def __init__(self, constituents: collections.abc.Iterable[BiltModelConstituent]) -> None:
        self.graph = rx.PyGraph(multigraph=False)
        self.attribute_nodes: dict[str, int] = {}
        self.constituent_nodes: list[int] = []

        for constituent in constituents:
            self._add_constituent(constituent)

    def _add_attribute(self, attribute: str) -> int:
        """Return the existing node index, or create a new attribute node."""
        if attribute not in self.attribute_nodes:
            self.attribute_nodes[attribute] = self.graph.add_node(attribute)
        return self.attribute_nodes[attribute]

    def _add_constituent(self, constituent: BiltModelConstituent) -> None:
        match constituent:
            case BasicTransform():
                attributes = {
                    *constituent.__bilt_parameters__,
                    constituent.__bilt_name__,
                }
            case BasicEquation() | SubSystemConstituent():
                attributes = set(constituent.__bilt_parameters__)
            case _:
                msg = f"Unsupported constituent type: {type(constituent)}"
                raise NotImplementedError(msg)

        constituent_node = self.graph.add_node(constituent)
        self.constituent_nodes.append(constituent_node)

        for attribute in sorted(attributes):
            attribute_node = self._add_attribute(attribute)
            self.graph.add_edge(constituent_node, attribute_node, None)
