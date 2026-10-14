__all__ = ["build_matching_graph", "find_matching"]

import typing

import rustworkx as rx

from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
        FullIncidenceGraph,
    )


def build_matching_graph(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
) -> rx.PyGraph:
    """Build candidate assignments to unknown attributes.

    Transforms can only produce their declared output. Subsystems have
    one candidate per unknown port, pending child-route validation.
    """
    graph = incidence.graph.copy()

    for node in incidence.constituent_nodes:
        constituent = incidence.graph[node]

        match constituent:
            case BasicTransform():
                for neighbor in incidence.graph.neighbors(node):
                    if incidence.graph[neighbor] != constituent.__bilt_name__:
                        graph.remove_edge(node, neighbor)

            case SubSystemConstituent():
                graph.remove_node(node)

                for output in sorted(constituent.__bilt_parameters__):
                    if output in available:
                        continue

                    candidate = graph.add_node(PlanStep(operation=constituent, output=output))
                    graph.add_edge(
                        candidate,
                        incidence.attribute_nodes[output],
                        None,
                    )

    for attribute in available:
        node = incidence.attribute_nodes.get(attribute)
        if node is not None:
            graph.remove_node(node)

    return graph


def find_matching(graph: rx.PyGraph) -> dict[str, BiltModelConstituent]:
    """Return a maximum candidate assignment, pending validation."""
    matching = rx.max_weight_matching(graph, max_cardinality=True)
    assignments: dict[str, BiltModelConstituent] = {}

    for first, second in matching:
        if isinstance(graph[first], str):
            attribute_node, constituent_node = first, second
        else:
            constituent_node, attribute_node = first, second

        attribute = typing.cast("str", graph[attribute_node])
        candidate = graph[constituent_node]

        constituent = (
            candidate.operation if isinstance(candidate, PlanStep) else typing.cast("BiltModelConstituent", candidate)
        )
        assignments[attribute] = constituent

    return assignments
