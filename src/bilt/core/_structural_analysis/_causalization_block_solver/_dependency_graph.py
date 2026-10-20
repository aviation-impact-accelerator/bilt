__all__ = ["build_dependency_graph"]

import typing

import rustworkx as rx

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._structural_analysis._plan import PlanStep


def build_dependency_graph(
    steps: collections.abc.Collection[PlanStep],
    available: collections.abc.Set[str],
) -> tuple[rx.PyDiGraph, dict[str, set[str]]]:
    """Build producer-to-consumer dependencies from settled steps.

    Node payloads are PlanSteps. Unresolved inputs are grouped by the
    output that requires them. Cycles are retained for SCC analysis.
    """
    graph = rx.PyDiGraph(multigraph=False)
    producer_nodes = {step.output: graph.add_node(step) for step in steps}
    unresolved: dict[str, set[str]] = {}

    for consumer in graph.node_indices():
        step = typing.cast("PlanStep", graph[consumer])

        for attribute in step.consumed:
            if attribute in available:
                continue

            producer = producer_nodes.get(attribute)
            if producer is None:
                unresolved.setdefault(step.output, set()).add(attribute)
            else:
                graph.add_edge(producer, consumer, None)

    return graph, unresolved
