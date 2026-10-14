__all__ = ["build_matching_graph", "find_matching"]

import typing

import rustworkx as rx

from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
        FullIncidenceGraph,
    )


def find_matching(graph: rx.PyGraph) -> dict[str, BiltModelConstituent]:
    """Assign as many unknown attributes as possible to distinct constituents.

    Returns a candidate assignment, which may be incomplete and still
    requires dependency and realisation checks.
    """
    matching = rx.max_weight_matching(graph, max_cardinality=True)

    assignments: dict[str, BiltModelConstituent] = {}
    for first, second in matching:
        if isinstance(graph[first], str):
            attribute_node, constituent_node = first, second
        else:
            constituent_node, attribute_node = first, second

        attribute = typing.cast("str", graph[attribute_node])
        constituent = typing.cast("BiltModelConstituent", graph[constituent_node])
        assignments[attribute] = constituent

    return assignments


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
