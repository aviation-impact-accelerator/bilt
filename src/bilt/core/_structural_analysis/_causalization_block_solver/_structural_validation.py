import typing

from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)

from ._dm import DMPartition, dulmage_mendelsohn

if typing.TYPE_CHECKING:
    import rustworkx as rx


def validate_structure(
    graph: rx.PyGraph,
    attribute_nodes: frozenset[int],
    input_nodes: frozenset[int],
) -> DMPartition:
    """Validate a scoped constraint graph after fixing supplied inputs.

    The caller establishes the constraint scope and resolves alternative
    implementations before calling this function.
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

    if partition.underdetermined:
        attributes = sorted(str(reduced[node]) for node in partition.underdetermined & unknown_nodes)
        message = f"Underconstrained problem involving attributes: {', '.join(attributes)}."
        raise BiltUnderconstrainedError(message)

    return partition
