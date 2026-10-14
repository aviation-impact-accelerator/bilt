__all__ = ["build_dependency_graph"]

import typing

import rustworkx as rx

from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent


def build_dependency_graph(
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    available: collections.abc.Set[str],
) -> tuple[rx.PyDiGraph, dict[str, set[str]]]:
    """Build producer-to-consumer dependencies for candidate assignments.

    Node payloads are PlanSteps. Unresolved inputs are grouped by the
    output that requires them. Cycles are retained for SCC analysis.
    """
    graph = rx.PyDiGraph(multigraph=False)
    producer_nodes: dict[str, int] = {}

    for output, constituent in assignments.items():
        if isinstance(constituent, SubSystemConstituent):
            msg = "Subsystem dependency analysis requires a selected child route."
            raise NotImplementedError(msg)

        step = PlanStep(operation=constituent, output=output)
        producer_nodes[output] = graph.add_node(step)

    unresolved: dict[str, set[str]] = {}

    for output, consumer in producer_nodes.items():
        step = typing.cast("PlanStep", graph[consumer])

        for attribute in step.consumed:
            if attribute in available:
                continue

            producer = producer_nodes.get(attribute)
            if producer is None:
                unresolved.setdefault(output, set()).add(attribute)
            else:
                graph.add_edge(producer, consumer, None)

    return graph, unresolved
