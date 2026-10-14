__all__ = ["build_matching_graph"]

import typing

from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
        FullIncidenceGraph,
    )


def build_matching_graph(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
) -> rx.PyGraph:
    """Build candidate assignments to unknown attributes.

    Transforms can only be assigned to their declared output.
    Equation and subsystem assignments still require realisation checks.
    """
    graph = incidence.graph.copy()

    for node in incidence.constituent_nodes:
        constituent = incidence.graph[node]
        if isinstance(constituent, BasicTransform):
            for neighbor in incidence.graph.neighbors(node):
                if incidence.graph[neighbor] != constituent.__bilt_name__:
                    graph.remove_edge(node, neighbor)

    for attribute in available:
        node = incidence.attribute_nodes.get(attribute)
        if node is not None:
            graph.remove_node(node)

    return graph
