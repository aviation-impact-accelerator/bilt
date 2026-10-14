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
    graph, unresolved = build_dependency_graph(
        {"c": addition, "d": d},
        {"a", "b"},
    )

    assert set(graph.nodes()) == {
        PlanStep(operation=addition, output="c"),
        PlanStep(operation=d, output="d"),
    }
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {("c", "d")}
    assert unresolved == {}


def test_dependency_graph_handles_branching_and_joining() -> None:
    graph, unresolved = build_dependency_graph(
        {"c": addition, "d": d, "e": e, "f": f},
        {"a", "b"},
    )

    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("c", "d"),
        ("c", "e"),
        ("d", "f"),
        ("e", "f"),
    }
    assert unresolved == {}


def test_dependency_graph_keeps_independent_computations() -> None:
    graph, unresolved = build_dependency_graph(
        {"d": d, "e": e},
        {"c"},
    )

    assert {step.output for step in graph.nodes()} == {"d", "e"}
    assert graph.num_edges() == 0
    assert unresolved == {}


def test_dependency_graph_uses_selected_equation_output() -> None:
    graph, unresolved = build_dependency_graph(
        {"b": addition},
        {"a", "c"},
    )

    assert list(graph.nodes()) == [
        PlanStep(operation=addition, output="b"),
    ]
    assert graph.num_edges() == 0
    assert unresolved == {}


def test_dependency_graph_preserves_cycles() -> None:
    graph, unresolved = build_dependency_graph(
        {"x": x, "y": y},
        set(),
    )

    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("x", "y"),
        ("y", "x"),
    }
    assert unresolved == {}


def test_dependency_graph_reports_unresolved_inputs() -> None:
    graph, unresolved = build_dependency_graph(
        {"c": addition, "d": d},
        {"a"},
    )

    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {("c", "d")}
    assert unresolved == {"c": {"b"}}


def test_empty_assignment_has_no_dependencies() -> None:
    graph, unresolved = build_dependency_graph({}, set())

    assert graph.num_nodes() == 0
    assert graph.num_edges() == 0
    assert unresolved == {}
