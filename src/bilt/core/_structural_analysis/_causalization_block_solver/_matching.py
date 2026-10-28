__all__ = ["build_matching_graph", "find_matching", "prefer_structural_producers"]

import typing

import rustworkx as rx

from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
        ExpandedConstraint,
        ExpansionSnapshot,
    )
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
        attribute_node = incidence.attribute_nodes.get(attribute)
        if attribute_node is not None:
            graph.remove_node(attribute_node)

    _prefer_available_inputs(graph, available)
    return graph


def _prefer_available_inputs(graph: rx.PyGraph, available: collections.abc.Set[str]) -> None:
    """Prefer primitive directions with supplied inputs without deleting alternatives."""
    for first, second in graph.edge_list():
        attribute, operation = (first, second) if isinstance(graph[first], str) else (second, first)
        candidate = graph[operation]
        if isinstance(candidate, PlanStep):
            weight = 0
        else:
            step = PlanStep(
                operation=typing.cast("BiltModelConstituent", candidate),
                output=typing.cast("str", graph[attribute]),
            )
            weight = int(set(step.consumed) <= available)
        graph.update_edge(first, second, weight)


def prefer_structural_producers(graph: rx.PyGraph, structure: ExpansionSnapshot) -> None:
    """Prefer forward-reachable primitive directions and their child instances.

    Propagation ignores isolation and backend support: these are ordering hints,
    never proof of executability. All matching edges and cardinality priorities
    remain intact so realisation failures can still select other directions.
    """
    if not structure.complete:
        return
    producers = _forward_producers(structure)
    for first, second in graph.edge_list():
        attribute, operation = (first, second) if isinstance(graph[first], str) else (second, first)
        source = producers.get(typing.cast("str", graph[attribute]))
        if source is None:
            continue
        candidate = graph[operation]
        if isinstance(candidate, PlanStep):
            preferred = bool(source.path) and source.path[0] == candidate.operation.__bilt_name__
        else:
            preferred = not source.path and source.operation is candidate
        if preferred:
            graph.update_edge(first, second, 2)


def _forward_producers(structure: ExpansionSnapshot) -> dict[str, ExpandedConstraint]:
    """Propagate primitive directions without choosing any execution route."""
    reached = set(structure.available)
    producers: dict[str, ExpandedConstraint] = {}
    pending = list(structure.constraints)
    while pending:
        deferred: list[ExpandedConstraint] = []
        for constraint in pending:
            unknown = constraint.attributes - reached
            if not unknown:
                continue
            if len(unknown) != 1:
                deferred.append(constraint)
                continue
            output = next(iter(unknown))
            if isinstance(constraint.operation, BasicTransform):
                declared = ".".join((*constraint.path, constraint.operation.__bilt_name__))
                inputs = {
                    ".".join((*constraint.path, parameter)) for parameter in constraint.operation.__bilt_parameters__
                }
                if output != declared or not inputs <= reached:
                    deferred.append(constraint)
                    continue
            producers[output] = constraint
            reached.add(output)
        if len(deferred) == len(pending):
            break
        pending = deferred

    return producers


def find_matching(graph: rx.PyGraph) -> dict[str, BiltModelConstituent]:
    """Maximise cardinality, using producer weights only to break ties.

    Weights prefer supplied inputs and forward-reachable structural producers.
    Other directions remain candidates for retries.
    """
    matching = rx.max_weight_matching(
        graph, max_cardinality=True, weight_fn=lambda weight: weight if isinstance(weight, int) else 0
    )
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
