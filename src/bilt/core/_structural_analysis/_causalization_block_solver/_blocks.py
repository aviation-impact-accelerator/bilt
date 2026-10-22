__all__ = ["ordered_blocks"]

import typing

import rustworkx as rx

from bilt.core._structural_analysis._plan import PlanBlock

if typing.TYPE_CHECKING:
    from bilt.core._structural_analysis._plan import PlanStep


def ordered_blocks(
    dependency_graph: rx.PyDiGraph,
) -> tuple[PlanBlock, ...]:
    """Group strongly connected steps and order producers before consumers.

    Steps within a block are sorted by output for reproducibility.
    That internal order is not an execution order for coupled blocks.
    """
    block_graph = rx.PyDiGraph(multigraph=False)
    node_to_block: dict[int, int] = {}

    for component in rx.strongly_connected_components(dependency_graph):
        steps = tuple(
            sorted(
                (typing.cast("PlanStep", dependency_graph[node]) for node in component),
                key=lambda step: step.output,
            )
        )
        block = PlanBlock(
            steps=steps,
            is_cyclic=len(component) > 1 or dependency_graph.has_edge(component[0], component[0]),
        )
        block_node = block_graph.add_node(block)

        for node in component:
            node_to_block[node] = block_node

    for source, target in dependency_graph.edge_list():
        source_block = node_to_block[source]
        target_block = node_to_block[target]

        if source_block != target_block:
            block_graph.add_edge(source_block, target_block, None)

    def block_key(block: PlanBlock) -> str:
        return block.steps[0].output

    return tuple(rx.lexicographical_topological_sort(block_graph, block_key))
