from bilt import equation, transform
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import (
    build_dependency_graph,
)
from bilt.core._structural_analysis._plan import PlanStep


@equation
def addition(a: float, b: float, c: float) -> bool:
    return a + b == c


@transform
def d(c: float) -> float:
    return c * 2


@transform
def e(c: float) -> float:
    return c + 1


@transform
def f(d: float, e: float) -> float:
    return d + e


@transform
def x(y: float) -> float:
    return y + 1


@transform
def y(x: float) -> float:
    return x / 2


def test_dependency_graph_connects_producer_to_consumer() -> None:
    steps = (
        PlanStep(operation=addition, output="c"),
        PlanStep(operation=d, output="d"),
    )

    graph, unresolved = build_dependency_graph(steps, {"a", "b"})

    assert set(graph.nodes()) == set(steps)
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {("c", "d")}
    assert unresolved == {}


def test_dependency_graph_handles_branching_and_joining() -> None:
    steps = (
        PlanStep(operation=addition, output="c"),
        PlanStep(operation=d, output="d"),
        PlanStep(operation=e, output="e"),
        PlanStep(operation=f, output="f"),
    )

    graph, unresolved = build_dependency_graph(steps, {"a", "b"})

    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("c", "d"),
        ("c", "e"),
        ("d", "f"),
        ("e", "f"),
    }
    assert unresolved == {}


def test_dependency_graph_keeps_independent_computations() -> None:
    steps = (
        PlanStep(operation=d, output="d"),
        PlanStep(operation=e, output="e"),
    )

    graph, unresolved = build_dependency_graph(steps, {"c"})

    assert set(graph.nodes()) == set(steps)
    assert graph.num_edges() == 0
    assert unresolved == {}


def test_dependency_graph_uses_selected_equation_output() -> None:
    step = PlanStep(operation=addition, output="b")

    graph, unresolved = build_dependency_graph((step,), {"a", "c"})

    assert list(graph.nodes()) == [step]
    assert graph.num_edges() == 0
    assert unresolved == {}


def test_dependency_graph_preserves_cycles() -> None:
    steps = (
        PlanStep(operation=x, output="x"),
        PlanStep(operation=y, output="y"),
    )

    graph, unresolved = build_dependency_graph(steps, set())

    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("x", "y"),
        ("y", "x"),
    }
    assert unresolved == {}


def test_dependency_graph_reports_unresolved_inputs() -> None:
    steps = (
        PlanStep(operation=addition, output="c"),
        PlanStep(operation=d, output="d"),
    )

    graph, unresolved = build_dependency_graph(steps, {"a"})

    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {("c", "d")}
    assert unresolved == {"c": {"b"}}


def test_empty_assignment_has_no_dependencies() -> None:
    graph, unresolved = build_dependency_graph((), set())

    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0
    assert unresolved == {}
