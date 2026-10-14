__all__ = ["DMPartition", "dulmage_mendelsohn"]

import dataclasses
import typing

import rustworkx as rx

if typing.TYPE_CHECKING:
    import collections.abc


@dataclasses.dataclass(frozen=True)
class DMPartition:
    """Coarse structural regions, expressed as indices in the input graph.

    Each region includes both attribute and operation nodes.
    These classifications describe the supplied graph, not numerical
    consistency, solvability or validity of subsystem routes.
    """

    underdetermined: frozenset[int]
    well_determined: frozenset[int]
    overdetermined: frozenset[int]


def dulmage_mendelsohn(
    graph: rx.PyGraph,
    attribute_nodes: collections.abc.Set[int],
) -> DMPartition:
    """Partition a bipartite graph using a maximum-cardinality matching.

    All nodes outside ``attribute_nodes`` belong to the operation side.
    The input graph is left unchanged.
    """
    nodes = set(graph.node_indices())
    attributes = set(attribute_nodes)

    if not attributes <= nodes:
        msg = "Attribute nodes must belong to the graph."
        raise ValueError(msg)

    edges = list(graph.edge_list())
    for first, second in edges:
        if (first in attributes) == (second in attributes):
            msg = "Every edge must connect an attribute to an operation."
            raise ValueError(msg)

    operations = nodes - attributes
    matching = rx.max_weight_matching(graph, max_cardinality=True)
    mates: dict[int, int] = {}

    for first, second in matching:
        mates[first] = second
        mates[second] = first

    forward: dict[int, list[int]] = {node: [] for node in nodes}
    reverse: dict[int, list[int]] = {node: [] for node in nodes}

    for first, second in edges:
        if first in attributes:
            attribute, operation = first, second
        else:
            operation, attribute = first, second

        # Starting at an unmatched attribute, alternate between
        # unmatched attribute -> operation edges and matched
        # operation -> attribute edges.
        if mates.get(attribute) == operation:
            source, target = operation, attribute
        else:
            source, target = attribute, operation

        forward[source].append(target)
        reverse[target].append(source)

    underdetermined = _reachable(
        forward,
        attributes - mates.keys(),
    )
    overdetermined = _reachable(
        reverse,
        operations - mates.keys(),
    )

    return DMPartition(
        underdetermined=frozenset(underdetermined),
        well_determined=frozenset(
            nodes - underdetermined - overdetermined,
        ),
        overdetermined=frozenset(overdetermined),
    )


def _reachable(
    adjacency: collections.abc.Mapping[int, collections.abc.Iterable[int]],
    sources: collections.abc.Iterable[int],
) -> set[int]:
    """Return sources and every node reachable from them."""
    reached = set(sources)
    pending = list(reached)

    while pending:
        for neighbor in adjacency[pending.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                pending.append(neighbor)

    return reached
