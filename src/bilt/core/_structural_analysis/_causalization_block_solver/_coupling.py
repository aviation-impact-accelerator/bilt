"""Recognise unavoidable coupling before enumerating execution candidates."""

__all__ = ["unavoidable_coupled_outputs"]

import typing

import rustworkx as rx

from bilt.core._structural_analysis._linear_block import (
    BiltLinearBlockError,
    extract_linear_block,
)
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep
from bilt.core._transform import BasicTransform

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
        ExpandedConstraint,
        ExpansionSnapshot,
    )


def unavoidable_coupled_outputs(
    structure: ExpansionSnapshot, output: str, *, allow_local_linear: bool = False
) -> frozenset[str]:
    """Return coupled ancestors of the target in a complete square region.

    Call after full-scope validation, using its well-determined region when the
    full structure is rectangular. With a perfect assignment of every
    primitive row and unknown, the irreducible equation blocks are invariant
    under alternative perfect matchings. Transforms are forced to their declared
    outputs. Their inputs remain dependencies, never alternative assignments.

    Incomplete, non-square, non-closed or directionally unmatchable regions are
    left to execution search. A cycle in an arbitrary partial assignment is not proof.

    With ``allow_local_linear``, omit components whose primitive equations can
    be extracted locally. Components involving expanded child paths remain
    unsupported until joint execution across subsystem boundaries is available.
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
    return _unsupported_outputs(dependencies, assigned, required, allow_local_linear=allow_local_linear)


def _unsupported_outputs(
    dependencies: rx.PyDiGraph,
    assigned: dict[str, ExpandedConstraint],
    required: set[int],
    *,
    allow_local_linear: bool,
) -> frozenset[str]:
    unsupported: set[str] = set()
    for component in rx.strongly_connected_components(dependencies):
        if len(component) <= 1 or not required.intersection(component):
            continue
        outputs = sorted(typing.cast("str", dependencies[node]) for node in component)
        if allow_local_linear and all(not assigned[name].path for name in outputs):
            block = PlanBlock(
                steps=tuple(PlanStep(operation=assigned[name].operation, output=name) for name in outputs),
                is_cyclic=True,
            )
            try:
                extract_linear_block(block)
            except BiltLinearBlockError:
                pass
            else:
                continue
        unsupported.update(outputs)
    return frozenset(unsupported)
