import rustworkx as rx

from bilt import transform
from bilt.core._structural_analysis._causalization_block_solver._blocks import (
    ordered_blocks,
)
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import (
    build_dependency_graph,
)
from bilt.core._structural_analysis._plan import PlanBlock, PlanStep


@transform
def b(a: float) -> float:
    return a * 2


@transform
def c(b: float) -> float:
    return b + 1


@transform
def d(b: float) -> float:
    return b + 2


@transform
def e(c: float, d: float) -> float:
    return c + d


@transform
def x(y: float, b: float) -> float:
    return y + b


@transform
def y(x: float) -> float:
    return x / 2


@transform
def z(x: float) -> float:
    return x + 1


def test_blocks_order_a_chain() -> None:
    first = PlanStep(operation=b, output="b")
    second = PlanStep(operation=c, output="c")
    graph, unresolved = build_dependency_graph(
        (second, first),
        {"a"},
    )

    blocks = ordered_blocks(graph)

    assert unresolved == {}
    assert blocks == (PlanBlock(steps=(first,)), PlanBlock(steps=(second,)))
    assert all(not block.is_cyclic for block in blocks)
    assert blocks[0].inputs == frozenset({"a"})
    assert blocks[1].inputs == frozenset({"b"})


def test_blocks_order_independent_branches_deterministically() -> None:
    steps = (
        PlanStep(operation=e, output="e"),
        PlanStep(operation=d, output="d"),
        PlanStep(operation=c, output="c"),
        PlanStep(operation=b, output="b"),
    )
    graph, unresolved = build_dependency_graph(steps, {"a"})

    blocks = ordered_blocks(graph)

    assert unresolved == {}
    assert tuple(tuple(step.output for step in block.steps) for block in blocks) == (("b",), ("c",), ("d",), ("e",))


def test_blocks_group_a_fully_coupled_system() -> None:
    first = PlanStep(operation=x, output="x")
    second = PlanStep(operation=y, output="y")
    graph, unresolved = build_dependency_graph(
        (second, first),
        {"b"},
    )

    blocks = ordered_blocks(graph)

    assert unresolved == {}
    assert blocks == (PlanBlock(steps=(first, second), is_cyclic=True),)
    assert blocks[0].outputs == frozenset({"x", "y"})
    assert blocks[0].inputs == frozenset({"b"})
    assert blocks[0].operations == (x, y)


def test_blocks_order_a_coupled_group_between_direct_steps() -> None:
    steps = (
        PlanStep(operation=z, output="z"),
        PlanStep(operation=y, output="y"),
        PlanStep(operation=x, output="x"),
        PlanStep(operation=b, output="b"),
    )
    graph, unresolved = build_dependency_graph(steps, {"a"})

    blocks = ordered_blocks(graph)

    assert unresolved == {}
    assert tuple(tuple(step.output for step in block.steps) for block in blocks) == (("b",), ("x", "y"), ("z",))
    assert tuple(block.is_cyclic for block in blocks) == (False, True, False)
    assert tuple(block.inputs for block in blocks) == (frozenset({"a"}), frozenset({"b"}), frozenset({"x"}))


def test_blocks_preserve_original_step_objects() -> None:
    step = PlanStep(operation=b, output="b")
    graph, _ = build_dependency_graph((step,), {"a"})

    blocks = ordered_blocks(graph)

    assert blocks[0].steps[0] is step


def test_blocks_leave_dependency_graph_unchanged() -> None:
    steps = (
        PlanStep(operation=x, output="x"),
        PlanStep(operation=y, output="y"),
    )
    graph, _ = build_dependency_graph(steps, {"b"})
    original_nodes = list(graph.nodes())
    original_edges = set(graph.edge_list())

    ordered_blocks(graph)

    assert list(graph.nodes()) == original_nodes
    assert set(graph.edge_list()) == original_edges


def test_blocks_include_a_singleton_with_a_self_dependency() -> None:
    graph = rx.PyDiGraph(multigraph=False)
    step = PlanStep(operation=b, output="b")
    node = graph.add_node(step)
    graph.add_edge(node, node, None)

    blocks = ordered_blocks(graph)

    assert blocks == (PlanBlock(steps=(step,), is_cyclic=True),)
    assert graph.has_edge(node, node)


def test_empty_graph_has_no_blocks() -> None:
    graph = rx.PyDiGraph(multigraph=False)

    assert ordered_blocks(graph) == ()
