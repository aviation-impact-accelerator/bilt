__all__ = ["build_matching_graph", "find_matching", "prefer_structural_producers"]

import typing

import rustworkx as rx

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._linear_operator import (
    extract_supported_linear_block,
    extract_supported_qualified_block,
)
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
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
        node = incidence.attribute_nodes.get(attribute)
        if node is not None:
            graph.remove_node(node)

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
            coupled = _coupled_linear_producers(deferred, reached)
            if not coupled:
                break
            producers.update(coupled)
            reached.update(coupled)
        pending = deferred

    return producers


def _coupled_linear_producers(
    constraints: list[ExpandedConstraint], reached: set[str]
) -> dict[str, ExpandedConstraint]:
    """Prefer ready, supported linear blocks of any size.

    Use a maximum matching and its dependency cycles rather than enumerating
    combinations of equation rows. Try child-local groups before mixed owners.
    These are only preference hints: normal planning still validates execution.
    """
    by_path: dict[tuple[str, ...], list[ExpandedConstraint]] = {}
    for constraint in constraints:
        if isinstance(constraint.operation, BasicEquation) and len(constraint.attributes - reached) > 1:
            by_path.setdefault(constraint.path, []).append(constraint)

    for path in sorted(by_path, key=lambda item: (-len(item), item)):
        preferred = _ready_linear_component(by_path[path], reached)
        if preferred:
            return preferred

    if len(by_path) > 1:
        return _ready_linear_component([row for group in by_path.values() for row in group], reached)
    return {}


def _ready_linear_component(constraints: list[ExpandedConstraint], reached: set[str]) -> dict[str, ExpandedConstraint]:
    """Find a supported dependency closure containing a coupled component."""
    if len(constraints) < 2:  # noqa: PLR2004 - coupled blocks require multiple rows
        return {}

    assigned = _matched_linear_constraints(constraints, reached)
    dependencies = rx.PyDiGraph(multigraph=False)
    nodes = {name: dependencies.add_node(name) for name in sorted(assigned)}
    for output, constraint in assigned.items():
        for name in sorted(constraint.attributes - reached - {output}):
            if name in nodes:
                dependencies.add_edge(nodes[name], nodes[output], None)

    for component in rx.strongly_connected_components(dependencies):
        if len(component) < 2:  # noqa: PLR2004 - only coupled blocks need a hint
            continue
        closure = set(component)
        for node in component:
            closure.update(rx.ancestors(dependencies, node))
        selected = {
            typing.cast("str", dependencies[node]): assigned[typing.cast("str", dependencies[node])] for node in closure
        }
        if any(row.attributes - reached - selected.keys() for row in selected.values()):
            continue
        if _supported_linear_component(selected):
            return selected
    return {}


def _matched_linear_constraints(
    constraints: list[ExpandedConstraint], reached: set[str]
) -> dict[str, ExpandedConstraint]:
    """Choose one maximum matching in the unresolved equation incidence graph."""
    incidence = rx.PyGraph(multigraph=False)
    names = sorted({name for row in constraints for name in row.attributes - reached})
    attribute_nodes = {name: incidence.add_node(name) for name in names}
    for constraint in constraints:
        row = incidence.add_node(constraint)
        for name in sorted(constraint.attributes - reached):
            incidence.add_edge(row, attribute_nodes[name], None)

    assignments: dict[str, ExpandedConstraint] = {}
    for first, second in rx.max_weight_matching(incidence, max_cardinality=True):
        row, attribute = (second, first) if isinstance(incidence[first], str) else (first, second)
        assignments[typing.cast("str", incidence[attribute])] = typing.cast("ExpandedConstraint", incidence[row])
    return assignments


def _supported_linear_component(selected: dict[str, ExpandedConstraint]) -> bool:
    """Validate linearity using the existing local or qualified block machinery."""
    outputs = sorted(selected)
    paths = {row.path for row in selected.values()}
    try:
        if len(paths) == 1:
            path = next(iter(paths))
            prefix = ".".join((*path, "")) if path else ""
            if any(not name.startswith(prefix) for name in outputs):
                return False
            block = PlanBlock(
                steps=tuple(
                    PlanStep(operation=selected[name].operation, output=name[len(prefix) :]) for name in outputs
                ),
                is_cyclic=True,
            )
            extract_supported_linear_block(block)
        else:
            block = QualifiedBlock(
                rows=tuple(
                    QualifiedEquation(typing.cast("BasicEquation", selected[name].operation), selected[name].path)
                    for name in outputs
                ),
                unknowns=tuple(outputs),
            )
            extract_supported_qualified_block(block)
    except BiltAnalysisError:
        return False
    return True


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
