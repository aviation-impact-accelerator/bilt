import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import (
    build_dependency_graph,
)
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import (
    settle_assignments,
)
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent


@system
class Child(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@pytest.fixture
def child() -> SubSystemConstituent:
    @transform
    def b(a: float) -> float:
        return a * 2

    @equation
    def relation(b: float, c: float) -> bool:
        return c == b + 1

    return SubSystemConstituent(
        name="child",
        implementation=Implementation(Child, [b, relation]),
    )


def test_settlement_preserves_equation_and_transform_assignments() -> None:
    @equation
    def relation(a: float, b: float) -> bool:
        return b == a + 1

    @transform
    def c(b: float) -> float:
        return b * 2

    steps, rejected = settle_assignments(
        {"b": relation, "c": c},
        {"a"},
        PYTHON_BACKEND,
    )

    assert set(steps) == {
        PlanStep(operation=relation, output="b"),
        PlanStep(operation=c, output="c"),
    }
    assert rejected == frozenset()


@pytest.mark.parametrize("input_port", ("a", "b"))
def test_settlement_selects_a_route_from_offered_child_ports(
    child: SubSystemConstituent,
    input_port: str,
) -> None:
    steps, rejected = settle_assignments(
        {"child.c": child},
        {f"child.{input_port}", "unrelated"},
        PYTHON_BACKEND,
    )

    assert len(steps) == 1
    assert steps[0].operation is child
    assert steps[0].output == "child.c"
    assert steps[0].child_inputs == frozenset({input_port})
    assert steps[0].consumed == (f"child.{input_port}",)
    assert rejected == frozenset()


def test_settlement_accepts_a_parent_producer_for_a_child_input(
    child: SubSystemConstituent,
) -> None:
    @equation
    def supply(source: float, child: Child) -> bool:
        return child.a == source

    steps, rejected = settle_assignments(
        {"child.a": supply, "child.c": child},
        {"source"},
        PYTHON_BACKEND,
    )
    graph, unresolved = build_dependency_graph(steps, {"source"})

    assert rejected == frozenset()
    assert unresolved == {}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("child.a", "child.c")
    }


def test_settlement_supports_multiple_child_outputs(
    child: SubSystemConstituent,
) -> None:
    steps, rejected = settle_assignments(
        {"child.b": child, "child.c": child},
        {"child.a"},
        PYTHON_BACKEND,
    )
    by_output = {step.output: step for step in steps}
    graph, unresolved = build_dependency_graph(steps, {"child.a"})

    assert set(by_output) == {"child.b", "child.c"}
    assert by_output["child.b"].child_inputs == frozenset({"a"})
    assert by_output["child.c"].child_inputs == frozenset({"b"})
    assert rejected == frozenset()
    assert unresolved == {}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("child.b", "child.c")
    }


def test_settlement_reports_an_unavailable_child_route(
    child: SubSystemConstituent,
) -> None:
    steps, rejected = settle_assignments(
        {"child.c": child},
        {"unrelated"},
        PYTHON_BACKEND,
    )

    assert steps == ()
    assert rejected == frozenset({"child.c"})


def test_settlement_connects_sibling_subsystems(
    child: SubSystemConstituent,
) -> None:
    left = SubSystemConstituent("left", child.implementation)
    right = SubSystemConstituent("right", child.implementation)

    @equation
    def connection(left: Child, right: Child) -> bool:
        return right.a == left.c

    steps, rejected = settle_assignments(
        {
            "left.c": left,
            "right.a": connection,
            "right.c": right,
        },
        {"left.a"},
        PYTHON_BACKEND,
    )
    graph, unresolved = build_dependency_graph(steps, {"left.a"})

    assert rejected == frozenset()
    assert unresolved == {}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("left.c", "right.a"),
        ("right.a", "right.c"),
    }


def test_settlement_recurses_through_nested_subsystems(
    child: SubSystemConstituent,
) -> None:
    @system
    class Outer(metaclass=Interface):
        source: float = field()
        result: float = field()

    @equation
    def supply(source: float, inner: Child) -> bool:
        return inner.a == source

    @equation
    def result_relation(result: float, inner: Child) -> bool:
        return result == inner.c

    outer = SubSystemConstituent(
        name="outer",
        implementation=Implementation(
            Outer,
            [supply, result_relation],
            internal_implementations={"inner": child.implementation},
        ),
    )

    steps, rejected = settle_assignments(
        {"outer.result": outer},
        {"outer.source"},
        PYTHON_BACKEND,
    )

    assert len(steps) == 1
    assert steps[0].child_inputs == frozenset({"source"})
    assert steps[0].consumed == ("outer.source",)
    assert rejected == frozenset()


def test_settlement_preserves_feedback_across_a_child_boundary(
    child: SubSystemConstituent,
) -> None:
    @equation
    def feedback(child: Child) -> bool:
        return child.a == child.c / 4

    steps, rejected = settle_assignments(
        {"child.a": feedback, "child.c": child},
        set(),
        PYTHON_BACKEND,
    )
    graph, unresolved = build_dependency_graph(steps, set())

    assert rejected == frozenset()
    assert unresolved == {}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("child.a", "child.c"),
        ("child.c", "child.a"),
    }
