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
) -> ExpansionSnapshot | None:
    """Expand relevant children structurally, then validate the combined scope.

    Unresolved child boundaries are not independent equation rows. Expand every
    boundary reached in the unknown request scope before definitive DM analysis.
    Parent and child primitives remain simultaneous constraints; execution choices
    and previous compilation failures do not affect this graph. Return the
    validated snapshot for execution preflight, or None for a supplied target.
    """
    if output in available:
        return None

    expansion = StructuralExpansion(incidence, available, output)
    graph, attributes = expansion.collect()
    try:
        validate_structure(
            graph,
            frozenset(attributes.values()),
            frozenset(),
            required_nodes=frozenset({attributes[output]}),
        )
    except (BiltUnderconstrainedError, BiltOverconstrainedError) as error:
        message = f"Cannot compute {output} from available inputs {sorted(available)}. {error}"
        raise type(error)(message) from error

    return expansion.snapshot()
