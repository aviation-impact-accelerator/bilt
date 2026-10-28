"""Recognise unavoidable coupling before enumerating execution candidates."""

__all__ = ["unavoidable_coupled_outputs"]

import typing

import rustworkx as rx

from bilt.core._equation import BasicEquation
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError, extract_qualified_linear_block
from bilt.core._structural_analysis._linear_operator import extract_supported_linear_block
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
        ExpandedConstraint,
        ExpansionSnapshot,
    )


def unavoidable_coupled_outputs(
    structure: ExpansionSnapshot,
    output: str,
    *,
    allow_local_linear: bool = False,
    allow_cross_boundary_linear: bool = False,
) -> frozenset[str]:
    """Return coupled ancestors of the target in a complete square region.

    Call after full-scope validation, using its well-determined region when the
    full structure is rectangular. With a perfect assignment of every
    primitive row and unknown, the irreducible equation blocks are invariant
    under alternative perfect matchings. Transforms are forced to their declared
    outputs. Their inputs remain dependencies, never alternative assignments.

    Incomplete, non-square, non-closed or directionally unmatchable regions are
    left to execution search. A cycle in an arbitrary partial assignment is not proof.

    With ``allow_local_linear``, omit components whose primitive equations all
    belong to one implementation and can be extracted in its local namespace.
    A child plans and realises its own blocks. ``allow_cross_boundary_linear``
    also permits primitive affine blocks spanning implementation paths; their
    input domains and backend capabilities remain realisation-time checks.
    """
    unknowns = structure.attributes - structure.available
    if not structure.complete or output not in unknowns or len(unknowns) != len(structure.constraints):
        return frozenset()

    if any(constraint.attributes - structure.available - unknowns for constraint in structure.constraints):
        return frozenset()

    graph = rx.PyGraph(multigraph=False)
    attributes = {name: graph.add_node(name) for name in sorted(unknowns)}
    for constraint in structure.constraints:
        row = graph.add_node(constraint)
        if isinstance(constraint.operation, BasicTransform):
            prefix = ".".join((*constraint.path, "")) if constraint.path else ""
            candidates = {prefix + constraint.operation.__bilt_name__}
        else:
            candidates = constraint.attributes
        for attribute in sorted(candidates & unknowns):
            graph.add_edge(row, attributes[attribute], None)

    matching = rx.max_weight_matching(graph, max_cardinality=True)
    if len(matching) != len(unknowns):
        return frozenset()

    dependencies = rx.PyDiGraph(multigraph=False)
    nodes = {name: dependencies.add_node(name) for name in sorted(unknowns)}
    assigned: dict[str, ExpandedConstraint] = {}
    for first, second in matching:
        row, attribute = (second, first) if isinstance(graph[first], str) else (first, second)
        constraint = typing.cast("ExpandedConstraint", graph[row])
        produced = typing.cast("str", graph[attribute])
        assigned[produced] = constraint
        for consumed in sorted((constraint.attributes & unknowns) - {produced}):
            dependencies.add_edge(nodes[consumed], nodes[produced], None)

    required = rx.ancestors(dependencies, nodes[output]) | {nodes[output]}
    return _unsupported_outputs(
        dependencies,
        assigned,
        required,
        allow_local_linear=allow_local_linear,
        allow_cross_boundary_linear=allow_cross_boundary_linear,
    )


def _unsupported_outputs(
    dependencies: rx.PyDiGraph,
    assigned: dict[str, ExpandedConstraint],
    required: set[int],
    *,
    allow_local_linear: bool,
    allow_cross_boundary_linear: bool,
) -> frozenset[str]:
    unsupported: set[str] = set()
    for component in rx.strongly_connected_components(dependencies):
        if len(component) <= 1 or not required.intersection(component):
            continue
        outputs = sorted(typing.cast("str", dependencies[node]) for node in component)
        if allow_local_linear and _locally_supported(outputs, assigned):
            continue
        if allow_cross_boundary_linear and _qualified_supported(outputs, assigned):
            continue
        unsupported.update(outputs)
    return frozenset(unsupported)


def _locally_supported(outputs: list[str], assigned: dict[str, ExpandedConstraint]) -> bool:
    """Check a single owner's equations using child-local output names."""
    path = assigned[outputs[0]].path
    if any(assigned[name].path != path for name in outputs):
        return False
    prefix = ".".join((*path, "")) if path else ""
    if any(not name.startswith(prefix) for name in outputs):
        return False
    block = PlanBlock(
        steps=tuple(PlanStep(operation=assigned[name].operation, output=name[len(prefix) :]) for name in outputs),
        is_cyclic=True,
    )
    try:
        extract_supported_linear_block(block)
    except BiltLinearBlockError:
        return False
    return True


def _qualified_supported(outputs: list[str], assigned: dict[str, ExpandedConstraint]) -> bool:
    if not all(isinstance(assigned[name].operation, BasicEquation) for name in outputs):
        return False
    try:
        block = QualifiedBlock(
            tuple(
                QualifiedEquation(typing.cast("BasicEquation", assigned[name].operation), assigned[name].path)
                for name in outputs
            ),
            tuple(outputs),
        )
        extract_qualified_linear_block(block)
    except BiltLinearBlockError:
        return False
    return True
