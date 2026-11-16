"""Discover request-scoped linear blocks from scalar-coordinate incidence.

This is an optional structural refinement of the existing port-level
matching/DM/SCC path. It operates on already expanded primitive equations,
uses exact named-domain declarations, and never executes model equations.
"""

__all__ = ["CoordinateBlockDiscovery", "discover_coordinate_blocks"]

import collections
import dataclasses
import inspect
import typing

import rustworkx as rx

from bilt.core._equation import BasicEquation
from bilt.core._errors import BiltDomainBindingError
from bilt.core._port_declaration import annotated_parts, quantity_magnitude_kind
from bilt.core._structural_analysis._causalization_block_solver._coordinate_incidence import (
    CoordinateIncidence,
    ResidualCoordinate,
    UnknownCoordinate,
    _CoordinateTracer,
    build_coordinate_incidence,
)
from bilt.core._structural_analysis._causalization_block_solver._dm import dulmage_mendelsohn
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._linear_block import BiltLinearBlockError
from bilt.core._structural_analysis._linear_domains import merge_domain_hints
from bilt.core._structural_analysis._qualified_block import QualifiedBlock, QualifiedEquation
from bilt.symbolic._algebra._affine import AffineAnalysisError
from bilt.symbolic._atoms import ISEL, MEAN, SEL, SUM, SymbolType
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from bilt.core._domain_registry import DomainRegistry
    from bilt.core._structural_analysis._causalization_block_solver._expand_structure import ExpansionSnapshot
    from bilt.symbolic._tree import Tree


@dataclasses.dataclass(frozen=True)
class CoordinateBlockDiscovery:
    """A verified structural partition, not a guarantee of numerical rank."""

    blocks: tuple[QualifiedBlock, ...]
    unknown_domains: tuple[tuple[str, Domain], ...]
    requires_execution: bool = False


def discover_coordinate_blocks(  # noqa: C901, PLR0911 - request validation dispatch.
    structure: ExpansionSnapshot,
    input_domains: Mapping[str, Domain],
    bindings: DomainRegistry,
    output: str,
) -> CoordinateBlockDiscovery | None:
    """Find complete linear blocks when all relevant domains are resolved.

    Return None for incomplete, unsupported, or ambiguous layouts, preserving
    the existing analyser's alternative route search. This refinement never
    interprets a minimum-domain quantity declaration as an exact tensor layout.
    It deliberately requires an operator; ordinary pointwise requests continue
    through the existing inexpensive port matching.
    """
    if not structure.complete or output in structure.available:
        return None
    if not structure.constraints or any(not isinstance(row.operation, BasicEquation) for row in structure.constraints):
        return None
    equations = tuple(
        QualifiedEquation(typing.cast("BasicEquation", row.operation), row.path) for row in structure.constraints
    )
    unknowns = structure.attributes - structure.available
    # A vector equality may determine several scalar ports without containing
    # an explicit reduction or selection operator.
    has_domain_operator = any(_has_operator(equation.tree) for equation in equations)
    multi_port_vector = len(equations) != len(unknowns) and any(domain.dims for domain in input_domains.values())
    if not (has_domain_operator or multi_port_vector):
        return None

    if output not in unknowns:
        return None
    hints = merge_domain_hints(hint for equation in equations for hint in equation.domain_hints)
    resolved: dict[str, Domain] = {
        name: hints[name] if name in hints else Domain.scalar()
        for name in sorted(unknowns)
        if name in hints or _declared_scalar(name, equations)
    }

    # A single scalar-declared equation port can still lift pointwise over
    # dimensions retained by known tensor inputs (e.g. sum(time) over a
    # (scenario, time) tensor). Do not interpret its scalar annotation as an
    # exact coordinate count: the established pointwise route handles it.
    if len(equations) == len(unknowns) == 1 and all(not layout.dims for layout in resolved.values()):
        return None

    # A complete operator candidate must have layouts for every external input.
    parameters = {name for equation in equations for name in equation.parameters}
    known = {name: input_domains[name] for name in parameters & structure.available if name in input_domains}
    if parameters - unknowns - known.keys():
        return None

    # Derived output layouts follow exact defining expressions, not minimum
    # Domain requirements on quantity-only tensor declarations.
    _infer_defined_domains(equations, unknowns, resolved, known, bindings)
    if set(resolved) != unknowns:
        return None

    try:
        incidence = build_coordinate_incidence(tuple(e.tree for e in equations), resolved, known, bindings)
    except (AffineAnalysisError, BiltDomainBindingError, BiltLinearBlockError):
        # Inference, operator support and expansion limits are independent of
        # the existing planner: it may still find a valid execution route.
        return None

    graph, attribute_nodes = incidence.to_graph()
    partition = dulmage_mendelsohn(graph, attribute_nodes)
    if partition.overdetermined:
        labels = sorted(
            {
                equations[typing.cast("ResidualCoordinate", graph[node]).equation].label
                for node in partition.overdetermined - attribute_nodes
            }
        )
        message = f"Overconstrained coordinate-level linear equations: {', '.join(labels)}."
        raise BiltOverconstrainedError(message)

    requested = {
        node for node in attribute_nodes if isinstance(graph[node], UnknownCoordinate) and graph[node].name == output
    }
    if requested & partition.underdetermined:
        names = sorted(
            {typing.cast("UnknownCoordinate", graph[node]).name for node in partition.underdetermined & attribute_nodes}
        )
        message = f"Underconstrained coordinate-level unknowns: {', '.join(names)}."
        raise BiltUnderconstrainedError(message)

    if partition.underdetermined or len(incidence.residuals) != len(incidence.unknowns):
        # The port planner still handles other structurally determined regions.
        return None

    blocks = _coalesce_axis_provenance(
        _ordered_coordinate_blocks(incidence, equations, output, resolved),
        known,
    )
    requires_execution = any(
        len(block.rows) != len(block.unknowns)
        or collections.Counter(incidence.residual_domains[equations.index(row)] for row in block.rows)
        != collections.Counter(resolved[name] for name in block.unknowns)
        for block in blocks
    )
    return CoordinateBlockDiscovery(blocks, tuple(sorted(resolved.items())), requires_execution)


def _has_operator(tree: Tree) -> bool:
    return tree.head in (SUM, MEAN, ISEL, SEL) or any(_has_operator(child) for child in tree.args)


def _declared_scalar(name: str, equations: tuple[QualifiedEquation, ...]) -> bool:
    """Accept scalar-only ports without imposing a tensor domain on unions."""
    annotations = []
    for equation in equations:
        prefix = ".".join((*equation.path, "")) if equation.path else ""
        if not name.startswith(prefix):
            continue
        local = name[len(prefix) :]
        parameters = inspect.signature(equation.source._function, eval_str=True).parameters  # noqa: SLF001
        if local in parameters:
            base, _ = annotated_parts(parameters[local].annotation)
            annotations.append(base)
    return bool(annotations) and all(
        annotation is float or quantity_magnitude_kind(annotation) == "scalar" for annotation in annotations
    )


def _ordered_coordinate_blocks(  # noqa: C901, PLR0912, PLR0915 - coordinate matching and SCC projection.
    incidence: CoordinateIncidence,
    equations: tuple[QualifiedEquation, ...],
    output: str,
    resolved: Mapping[str, Domain],
) -> tuple[QualifiedBlock, ...]:
    """Match coordinates, coalesce their equation/port owners, then take SCCs.

    Coalescing is essential: one primitive tensor equality can contain many
    residual coordinates and one tensor unknown can span many scalar columns.
    A block may therefore contain different counts of equation and port objects.
    """
    graph, _unknown_nodes = incidence.to_graph()
    matching = rx.max_weight_matching(graph, max_cardinality=True)
    if len(matching) != len(incidence.residuals):
        message = "Coordinate block discovery requires a complete structural matching."
        raise BiltUnderconstrainedError(message)

    matched: dict[ResidualCoordinate, UnknownCoordinate] = {}
    for first, second in matching:
        left, right = graph[first], graph[second]
        residual, unknown = (left, right) if isinstance(left, ResidualCoordinate) else (right, left)
        matched[typing.cast("ResidualCoordinate", residual)] = typing.cast("UnknownCoordinate", unknown)

    residual_index = {row: index for index, row in enumerate(incidence.residuals)}
    coordinate_owner = {unknown: residual_index[row] for row, unknown in matched.items()}
    parent = list(range(len(incidence.residuals)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(first: int, second: int) -> None:
        parent[root(first)] = root(second)

    eq_owner: dict[int, int] = {}
    port_owner: dict[str, int] = {}
    for row in incidence.residuals:
        index = residual_index[row]
        port = matched[row].name
        if row.equation in eq_owner:
            join(index, eq_owner[row.equation])
        else:
            eq_owner[row.equation] = index
        if port in port_owner:
            join(index, port_owner[port])
        else:
            port_owner[port] = index

    groups = sorted({root(index) for index in parent})
    dependency = rx.PyDiGraph(multigraph=False)
    nodes = {group: dependency.add_node(group) for group in groups}
    for residual, unknown in incidence.edges:
        consumed = root(coordinate_owner[unknown])
        produced = root(residual_index[residual])
        if consumed != produced and not dependency.has_edge(nodes[consumed], nodes[produced]):
            dependency.add_edge(nodes[consumed], nodes[produced], None)

    selected = {nodes[root(coordinate_owner[unknown])] for unknown in incidence.unknowns if unknown.name == output}
    required = selected | {node for target in selected for node in rx.ancestors(dependency, target)}
    condensed = rx.PyDiGraph(multigraph=False)
    node_mapping = {}
    for component in rx.strongly_connected_components(dependency):
        if not set(component) & required:
            continue
        group_ids = {typing.cast("int", dependency[node]) for node in component}
        members = [row for row in incidence.residuals if root(residual_index[row]) in group_ids]
        row_ids = sorted({row.equation for row in members})
        ports = sorted({matched[row].name for row in members})
        block = QualifiedBlock(
            tuple(equations[index] for index in row_ids),
            tuple(ports),
            resolved_domains=tuple((name, resolved[name]) for name in ports),
        )
        node = condensed.add_node((ports[0], block))
        node_mapping.update(dict.fromkeys(component, node))

    for first, second in dependency.edge_list():
        if (
            first in node_mapping
            and second in node_mapping
            and node_mapping[first] != node_mapping[second]
            and not condensed.has_edge(node_mapping[first], node_mapping[second])
        ):
            condensed.add_edge(node_mapping[first], node_mapping[second], None)

    return tuple(item[1] for item in rx.lexicographical_topological_sort(condensed, lambda item: item[0]))


def _coalesce_axis_provenance(
    blocks: tuple[QualifiedBlock, ...],
    known: Mapping[str, Domain],
) -> tuple[QualifiedBlock, ...]:
    """Preserve the inputs that establish the extents of a tensor unknown.

    Coordinate matching may find two acyclic blocks even though the first
    block's residual eliminates an axis absent from its external inputs.
    Its numerical basis cannot infer that axis's extent from those inputs
    alone. When the immediately dependent block has an input carrying the
    missing axes, retain both equations in one qualified linear solve.

    This is domain provenance, not an operator-specific pattern or a
    numerical choice: unrelated or self-contained blocks remain separate.
    """
    ordered = list(blocks)
    index = 0
    while index + 1 < len(ordered):
        first, second = ordered[index : index + 2]
        first_axes = {dim for name in first.inputs if name in known for dim in known[name].dims}
        output_axes = {dim for _, layout in first.resolved_domains for dim in layout.dims}
        missing_axes = output_axes - first_axes
        second_axes = {dim for name in second.inputs if name in known for dim in known[name].dims}
        if missing_axes and missing_axes <= second_axes and set(first.unknowns) & second.inputs:
            joined = QualifiedBlock(
                first.rows + second.rows,
                first.unknowns + second.unknowns,
                resolved_domains=first.resolved_domains + second.resolved_domains,
            )
            ordered[index : index + 2] = [joined]
            index = max(index - 1, 0)
        else:
            index += 1
    return tuple(ordered)


def _infer_defined_domains(
    equations: tuple[QualifiedEquation, ...],
    unknowns: frozenset[str],
    resolved: dict[str, Domain],
    inputs: Mapping[str, Domain],
    bindings: DomainRegistry,
) -> None:
    """Infer layouts of directly defined outputs from fully resolved expressions.

    Only a bare output on one side of an equality qualifies, and the other
    side's domain must already be known. This never promotes a tensor's
    minimum-domain declaration into an assumed exact tensor layout.
    """
    pending = set(unknowns) - resolved.keys()
    while pending:
        progress = False
        tracer = _CoordinateTracer(inputs | resolved, frozenset(), bindings)
        for equation in equations:
            tree = equation.tree
            if not tree.is_equality:
                continue
            for side, other in ((tree.left, tree.right), (tree.right, tree.left)):
                if not side.is_leaf or side.head.atom_type is not SymbolType:
                    continue
                name = typing.cast("str", side.head.value)
                if name not in pending:
                    continue
                try:
                    inferred = tracer.domain(other)
                except BiltLinearBlockError:
                    continue
                resolved[name] = inferred
                pending.remove(name)
                progress = True
                break
        if not progress:
            return
