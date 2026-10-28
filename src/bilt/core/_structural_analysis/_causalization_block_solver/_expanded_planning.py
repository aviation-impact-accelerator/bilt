"""Plan the primitive scope when a coupled block crosses implementation paths."""

import typing

import rustworkx as rx

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError, extract_qualified_linear_block
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation, QualifiedStep
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
        ExpandedConstraint,
        ExpansionSnapshot,
    )


def matching_graph(structure: ExpansionSnapshot) -> rx.PyGraph:
    """Use primitive rows, retaining transform direction and source identity."""
    graph = rx.PyGraph(multigraph=False)
    unknowns = structure.attributes - structure.available
    attributes = {name: graph.add_node(name) for name in sorted(unknowns)}
    for constraint in structure.constraints:
        row = graph.add_node(constraint)
        prefix = ".".join((*constraint.path, "")) if constraint.path else ""
        candidates = (
            {prefix + constraint.operation.__bilt_name__}
            if isinstance(constraint.operation, BasicTransform)
            else constraint.attributes
        )
        for name in sorted(candidates & unknowns):
            graph.add_edge(row, attributes[name], None)
    return graph


def matching_step(graph: rx.PyGraph, first: int, second: int) -> QualifiedStep:
    attribute, row = (first, second) if isinstance(graph[first], str) else (second, first)
    constraint = typing.cast("ExpandedConstraint", graph[row])
    return QualifiedStep(constraint.operation, constraint.path, typing.cast("str", graph[attribute]))


def plan_blocks(
    assignments: collections.abc.Mapping[str, QualifiedStep], available: set[str], output: str
) -> tuple[QualifiedStep | QualifiedBlock, ...]:
    """Select the target closure and retain ordered pointwise equation SCCs."""
    selected = {}
    pending = [output]
    while pending:
        name = pending.pop()
        if name in available or name in selected:
            continue
        if name not in assignments:
            message = f"No expanded primitive producer for {name!r}."
            raise BiltExecutionPlanError(message)
        step = assignments[name]
        selected[name] = step
        pending.extend(step.inputs)
    graph = rx.PyDiGraph(multigraph=False)
    nodes = {name: graph.add_node(step) for name, step in sorted(selected.items())}
    for name, step in selected.items():
        for dependency in step.inputs - available:
            graph.add_edge(nodes[dependency], nodes[name], None)
    return _ordered_blocks(graph)


def _ordered_blocks(graph: rx.PyDiGraph) -> tuple[QualifiedStep | QualifiedBlock, ...]:
    condensed = rx.PyDiGraph(multigraph=False)
    mapping = {}
    for component in rx.strongly_connected_components(graph):
        steps = tuple(sorted((typing.cast("QualifiedStep", graph[node]) for node in component), key=lambda s: s.output))
        cyclic = len(component) > 1 or graph.has_edge(component[0], component[0])
        block = _linear_block(steps) if cyclic else steps[0]
        node = condensed.add_node((steps[0].output, block))
        mapping.update(dict.fromkeys(component, node))
    for source, target in graph.edge_list():
        if mapping[source] != mapping[target]:
            condensed.add_edge(mapping[source], mapping[target], None)
    return tuple(item[1] for item in rx.lexicographical_topological_sort(condensed, lambda item: item[0]))


def _linear_block(steps: tuple[QualifiedStep, ...]) -> QualifiedBlock:
    if not all(isinstance(step.source, BasicEquation) for step in steps):
        message = "Execution of coupled blocks is not yet supported for qualified transform feedback."
        raise BiltExecutionPlanError(message)
    block = QualifiedBlock(
        tuple(QualifiedEquation(typing.cast("BasicEquation", step.source), step.path) for step in steps),
        tuple(step.output for step in steps),
    )
    try:
        extract_qualified_linear_block(block)
    except BiltLinearBlockError as error:
        message = f"Execution of coupled blocks is not yet supported for this qualified formulation: {error}"
        raise BiltExecutionPlanError(message) from error
    return block


def required_inputs(blocks: tuple[QualifiedStep | QualifiedBlock, ...]) -> frozenset[str]:
    produced = {
        name for block in blocks for name in (block.unknowns if isinstance(block, QualifiedBlock) else (block.output,))
    }
    return frozenset(name for block in blocks for name in block.inputs if name not in produced)
