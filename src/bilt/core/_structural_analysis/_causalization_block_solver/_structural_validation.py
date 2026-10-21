import dataclasses
import typing

from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpansionSnapshot,
    StructuralExpansion,
)

from ._dm import DMPartition, dulmage_mendelsohn

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import (
        FullIncidenceGraph,
    )


@dataclasses.dataclass(frozen=True)
class ValidatedStructure:
    """The full validated scope and its well-determined execution-preflight region."""

    snapshot: ExpansionSnapshot
    well_determined: ExpansionSnapshot


def validate_structure(
    graph: rx.PyGraph,
    attribute_nodes: frozenset[int],
    input_nodes: frozenset[int],
    *,
    required_nodes: frozenset[int] | None = None,
) -> DMPartition:
    """Validate a scoped constraint graph after fixing supplied inputs.

    The caller establishes the constraint scope. By default every unknown
    attribute is required. With required_nodes, deficiencies unrelated to
    those requested attributes are retained in the result without raising.
    """
    unknown_nodes = attribute_nodes - input_nodes

    reduced = graph.copy()
    reduced.remove_nodes_from(list(attribute_nodes & input_nodes))

    partition = dulmage_mendelsohn(reduced, unknown_nodes)

    if partition.overdetermined:
        constituents = sorted(str(reduced[node]) for node in partition.overdetermined - unknown_nodes)
        message = (
            f"Overconstrained problem involving constituents: "
            f"{', '.join(constituents)}. Reassess the supplied constituents."
        )
        raise BiltOverconstrainedError(message)

    required = unknown_nodes if required_nodes is None else required_nodes - input_nodes
    if partition.underdetermined & required:
        attributes = sorted(str(reduced[node]) for node in partition.underdetermined & unknown_nodes)
        message = f"Underconstrained problem involving attributes: {', '.join(attributes)}."
        raise BiltUnderconstrainedError(message)

    return partition


def validate_request_structure(
    incidence: FullIncidenceGraph,
    available: collections.abc.Set[str],
    output: str,
) -> ValidatedStructure | None:
    """Expand relevant children structurally, then validate the combined scope.

    Unresolved child boundaries are not independent equation rows. Expand every
    boundary reached in the unknown request scope before definitive DM analysis.
    Parent and child primitives remain simultaneous constraints; execution choices
    and previous compilation failures do not affect this graph. Return the
    full snapshot and its well-determined region, or None for a supplied target.
    """
    if output in available:
        return None

    expansion = StructuralExpansion(incidence, available, output)
    graph, attributes = expansion.collect()
    try:
        partition = validate_structure(
            graph,
            frozenset(attributes.values()),
            frozenset(),
            required_nodes=frozenset({attributes[output]}),
        )
    except (BiltUnderconstrainedError, BiltOverconstrainedError) as error:
        message = f"Cannot compute {output} from available inputs {sorted(available)}. {error}"
        raise type(error)(message) from error

    snapshot = expansion.snapshot()
    attribute_nodes = frozenset(attributes.values())
    well_attributes = frozenset(name for name, node in attributes.items() if node in partition.well_determined)
    # to_incidence inserts primitive rows in snapshot order after the attributes.
    # Map by node identity, not by labels, which can be shared by distinct rows.
    row_nodes = (node for node in graph.node_indices() if node not in attribute_nodes)
    well_constraints = tuple(
        constraint
        for constraint, node in zip(snapshot.constraints, row_nodes, strict=True)
        if node in partition.well_determined
    )
    return ValidatedStructure(
        snapshot=snapshot,
        well_determined=dataclasses.replace(
            snapshot,
            attributes=well_attributes | snapshot.available,
            constraints=well_constraints,
        ),
    )
