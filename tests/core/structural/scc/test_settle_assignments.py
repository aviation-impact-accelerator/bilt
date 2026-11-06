import copy
import itertools
import sys
import typing

import pytest
import rustworkx as rx

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import (
    build_dependency_graph,
)
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import (
    settle_assignments,
)
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@system
class Child(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@pytest.fixture(autouse=True)
def _causalization_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "CausalizationStructuralAnalyzer", CausalizationStructuralAnalyzer)


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
        target="c",
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
        target="child.c",
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
        target="child.c",
    )
    graph, unresolved = build_dependency_graph(steps, {"source"})

    assert rejected == frozenset()
    assert unresolved == {}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("child.a", "child.c")
    }


def test_settlement_reports_an_unavailable_child_route(
    child: SubSystemConstituent,
) -> None:
    steps, rejected = settle_assignments(
        {"child.c": child},
        {"unrelated"},
        PYTHON_BACKEND,
        target="child.c",
    )

    assert steps == ()
    assert rejected == frozenset({"child.c"})


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
        target="outer.result",
    )

    assert len(steps) == 1
    assert steps[0].child_inputs == frozenset({"source"})
    assert steps[0].consumed == ("outer.source",)
    assert rejected == frozenset()


def test_target_settlement_skips_unrelated_children(
    child: SubSystemConstituent, monkeypatch: pytest.MonkeyPatch
) -> None:
    @transform
    def result(a: float) -> float:
        return a + 1

    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("Unrelated child outputs must not be queried.")

    monkeypatch.setattr(SubSystemConstituent, "required_inputs_for", forbidden)
    steps, rejected = settle_assignments({"result": result, "child.c": child}, {"a"}, PYTHON_BACKEND, target="result")
    assert tuple(step.output for step in steps) == ("result",)
    assert not rejected


def test_supplied_target_requires_no_settlement(child: SubSystemConstituent, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("Supplied targets need no child route query.")

    monkeypatch.setattr(SubSystemConstituent, "required_inputs_for", forbidden)
    assert settle_assignments({"child.c": child}, {"answer"}, PYTHON_BACKEND, target="answer") == ((), frozenset())


@pytest.mark.parametrize("reverse", (False, True))
def test_target_settlement_connects_parent_inputs_and_multiple_child_outputs(
    child: SubSystemConstituent, *, reverse: bool
) -> None:
    @transform
    def z(source: float) -> float:
        return source * 2

    @equation
    def supply(z: float, child: Child) -> bool:
        return child.a == z

    @equation
    def combined(answer: float, child: Child) -> bool:
        return answer == child.b + child.c

    assignments = {"answer": combined, "child.a": supply, "child.b": child, "child.c": child, "z": z}
    if reverse:
        assignments = dict(reversed(tuple(assignments.items())))
    steps, rejected = settle_assignments(assignments, {"source"}, PYTHON_BACKEND, target="answer")
    by_output = {step.output: step for step in steps}
    assert by_output["child.b"].child_inputs == frozenset({"a"})
    assert by_output["child.c"].child_inputs == frozenset({"b"})
    graph, unresolved = build_dependency_graph(steps, {"source"})
    assert not rejected
    assert not unresolved
    assert rx.is_directed_acyclic_graph(graph)
    assert set(by_output) == {"answer", "child.a", "child.b", "child.c", "z"}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("z", "child.a"),
        ("child.a", "child.b"),
        ("child.b", "child.c"),
        ("child.b", "answer"),
        ("child.c", "answer"),
    }


def test_target_settlement_preserves_boundary_feedback(child: SubSystemConstituent) -> None:
    @equation
    def feedback(child: Child) -> bool:
        return child.a == child.c / 4

    steps, rejected = settle_assignments(
        {"child.a": feedback, "child.c": child}, set(), PYTHON_BACKEND, target="child.c"
    )
    graph, unresolved = build_dependency_graph(steps, set())
    assert not rejected
    assert not unresolved
    assert not rx.is_directed_acyclic_graph(graph)
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("child.a", "child.c"),
        ("child.c", "child.a"),
    }


def test_target_settlement_can_supply_a_boundary_from_a_sibling(child: SubSystemConstituent) -> None:
    left = SubSystemConstituent("left", child.implementation)
    right = SubSystemConstituent("right", child.implementation)

    @equation
    def connect(left: Child, right: Child) -> bool:
        return right.a == left.c

    steps, rejected = settle_assignments(
        {"left.c": left, "right.a": connect, "right.c": right}, {"left.a"}, PYTHON_BACKEND, target="right.c"
    )
    graph, unresolved = build_dependency_graph(steps, {"left.a"})
    assert not rejected
    assert not unresolved
    assert rx.is_directed_acyclic_graph(graph)
    assert {step.output for step in steps} == {"left.c", "right.a", "right.c"}
    assert {(graph[source].output, graph[target].output) for source, target in graph.edge_list()} == {
        ("left.c", "right.a"),
        ("right.a", "right.c"),
    }


def test_target_settlement_preserves_missing_inputs() -> None:
    @transform
    def result(missing: float) -> float:
        return missing + 1

    steps, rejected = settle_assignments({"result": result}, set(), PYTHON_BACKEND, target="result")
    _, unresolved = build_dependency_graph(steps, set())
    assert not rejected
    assert unresolved == {"result": frozenset({"missing"})}


def test_target_settlement_handles_chains_longer_than_the_python_recursion_limit() -> None:
    @transform
    def value(previous: float) -> float:
        return previous + 1

    # Structural-only chain: compilation is intentionally outside this test.
    length = sys.getrecursionlimit() + 50
    assignments: dict[str, BiltModelConstituent] = {}
    for index in range(length):
        operation = copy.copy(value)
        operation.__bilt_name__ = f"v{index + 1}"
        operation.__bilt_parameters__ = (f"v{index}",)
        assignments[operation.__bilt_name__] = operation
    steps, rejected = settle_assignments(assignments, {"v0"}, PYTHON_BACKEND, target=f"v{length}")
    assert not rejected
    assert len(steps) == length


@pytest.mark.parametrize("order", tuple(itertools.permutations(("a", "b", "z"))))
def test_settlement_revisits_outputs_after_later_producers(order: tuple[str, ...]) -> None:
    @transform
    def z(source: float) -> float:
        return source * 2

    @transform
    def a(z: float) -> float:
        return z + 1

    @transform
    def b(a: float) -> float:
        return a + 1

    operations = {"a": a, "b": b, "z": z}
    steps, rejected = settle_assignments(
        {name: operations[name] for name in order}, {"source"}, PYTHON_BACKEND, target="b"
    )
    graph, unresolved = build_dependency_graph(steps, {"source"})
    assert rejected == frozenset()
    assert unresolved == {}
    assert rx.is_directed_acyclic_graph(graph)
    assert tuple(step.output for step in steps) == ("a", "b", "z")
