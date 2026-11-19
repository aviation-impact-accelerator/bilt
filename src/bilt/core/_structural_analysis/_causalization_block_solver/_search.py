"""Enumerate matching candidates and prune failed assignment choices."""

__all__ = ["candidate_assignments"]

import contextlib
import contextvars
import typing

from bilt.core._structural_analysis._causalization_block_solver._matching import find_matching
from bilt.core._structural_analysis._causalization_block_solver._route_cache import route_cache_generation
from bilt.core._structural_analysis._plan import PlanStep

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._realize import RealizationFailure


_deduplicate: contextvars.ContextVar[bool] = contextvars.ContextVar("bilt_matching_dedup", default=False)


@contextlib.contextmanager
def matching_dedup_scope() -> collections.abc.Generator[None]:
    """Experimentally suppress duplicate candidate routes within one search."""
    token = _deduplicate.set(True)
    try:
        yield
    finally:
        _deduplicate.reset(token)


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


def _target_matching_edges(
    graph: rx.PyGraph,
    edges: list[tuple[int, int]],
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    output: str,
) -> set[tuple[int, int]]:
    """Keep alternatives that can change the target's dependency route."""
    attribute_nodes = {
        typing.cast("str", graph[node]): node for node in graph.node_indices() if isinstance(graph[node], str)
    }
    matched_edges: dict[str, tuple[int, int]] = {}
    matched_operations: dict[int, str] = {}
    for first, second in edges:
        step = _edge_step(graph, first, second)
        if assignments.get(step.output) is not step.operation:
            continue
        matched_edges[step.output] = (first, second)
        operation_node = second if isinstance(graph[first], str) else first
        matched_operations[operation_node] = step.output

    relevant: set[tuple[int, int]] = set()
    pending = [output]
    visited: set[str] = set()
    while pending:
        attribute = pending.pop()
        if attribute in visited:
            continue
        visited.add(attribute)

        if edge := matched_edges.get(attribute):
            relevant.add(edge)
            operation = assignments[attribute]
            pending.extend(parameter for parameter in operation.__bilt_parameters__ if parameter != attribute)
            continue

        # If an input has no assignment, a candidate producer may be matched
        # elsewhere. Freeing that producer can make the input computable.
        node = attribute_nodes.get(attribute)
        if node is not None:
            pending.extend(
                matched_operations[neighbor] for neighbor in graph.neighbors(node) if neighbor in matched_operations
            )
    return relevant


def _alternatives(
    graph: rx.PyGraph,
    edges: list[tuple[int, int]],
    assignments: collections.abc.Mapping[str, BiltModelConstituent],
    output: str | None,
) -> collections.abc.Iterator[rx.PyGraph]:
    relevant = _target_matching_edges(graph, edges, assignments, output) if output is not None else None
    for first, second in edges:
        if relevant is not None and (first, second) not in relevant:
            continue
        step = _edge_step(graph, first, second)
        if assignments.get(step.output) is not step.operation:
            continue
        alternative = graph.copy()
        alternative.remove_edge(first, second)
        yield alternative


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
    yielded: set[tuple[frozenset[tuple[str, int]], frozenset[PlanStep], int]] = set()

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
        if _deduplicate.get():
            signature = (
                frozenset((name, id(operation)) for name, operation in assignments.items()),
                frozenset(failures),
                route_cache_generation(),
            )
            if signature not in yielded:
                yielded.add(signature)
                yield assignments
        else:
            yield assignments

        # Realisation may have added failures while the generator yielded.
        # Retry this graph with those assignments excluded first.
        if any(_edge_step(candidate_graph, first, second) in failures for first, second in edges):
            pending.append(candidate_graph)
            continue

        pending.extend(_alternatives(candidate_graph, edges, assignments, output))
