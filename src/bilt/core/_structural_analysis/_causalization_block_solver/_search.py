"""Enumerate matching candidates and prune failed assignment choices."""

__all__ = ["candidate_assignments"]

import typing

from bilt.core._structural_analysis._causalization_block_solver._matching import find_matching
from bilt.core._structural_analysis._plan import PlanStep

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._realize import RealizationFailure


def _edge_step(graph: rx.PyGraph, first: int, second: int) -> PlanStep:
    """Identify the computation represented by a matching edge."""
    if isinstance(graph[first], str):
        attribute_node, constituent_node = first, second
    else:
        constituent_node, attribute_node = first, second

    candidate = graph[constituent_node]
    if isinstance(candidate, PlanStep):
        return candidate

    return PlanStep(
        operation=typing.cast("BiltModelConstituent", candidate),
        output=typing.cast("str", graph[attribute_node]),
    )


def candidate_assignments(
    graph: rx.PyGraph,
    failures: collections.abc.Mapping[PlanStep, RealizationFailure],
    *,
    output: str | None,
) -> collections.abc.Iterator[dict[str, BiltModelConstituent]]:
    """Try matchings while excluding failed or unsuccessful choices.

    Failed realisations exclude individual assignments. Other rejected
    candidates branch by removing selected matching edges. An unavailable
    output must retain a producer edge; supplied outputs pass output=None.
    """
    output_node = (
        next((node for node in graph.node_indices() if isinstance(graph[node], str) and graph[node] == output), None)
        if output is not None
        else None
    )
    if output is not None and output_node is None:
        return

    pending = [graph]
    visited: set[frozenset[tuple[int, int]]] = set()

    while pending:
        candidate_graph = pending.pop()

        for first, second in list(candidate_graph.edge_list()):
            if _edge_step(candidate_graph, first, second) in failures:
                candidate_graph.remove_edge(first, second)

        # Descendants only remove edges, so a lost target producer cannot
        # reappear. A matching that merely omits the target is still retried.
        if output_node is not None and candidate_graph.degree(output_node) == 0:
            continue

        edges = list(candidate_graph.edge_list())
        signature = frozenset((min(first, second), max(first, second)) for first, second in edges)
        if signature in visited:
            continue
        visited.add(signature)

        assignments = find_matching(candidate_graph)
        yield assignments

        # Realisation may have added failures while the generator yielded.
        # Retry this graph with those assignments excluded first.
        if any(_edge_step(candidate_graph, first, second) in failures for first, second in edges):
            pending.append(candidate_graph)
            continue

        for first, second in edges:
            step = _edge_step(candidate_graph, first, second)
            if assignments.get(step.output) is not step.operation:
                continue

            alternative = candidate_graph.copy()
            alternative.remove_edge(first, second)
            pending.append(alternative)
