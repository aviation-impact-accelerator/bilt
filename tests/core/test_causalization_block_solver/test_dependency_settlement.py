import itertools

import pytest
import rustworkx as rx

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._port import mangle_name
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._dependency_graph import build_dependency_graph
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._settle_assignments import settle_assignments
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
    x: float = field()
    y: float = field()


@transform
def b(a: float) -> float:
    return 2 * a


@equation
def next_value(b: float, c: float) -> bool:
    return c == b + 1


@equation
def supply(source: float, child: Ports) -> bool:
    return child.a == source


@equation
def combined(answer: float, child: Ports) -> bool:
    return answer == child.b + child.c


@pytest.fixture(autouse=True)
def _causalization_children(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", CausalizationStructuralAnalyzer)


def _run(analyser: CausalizationStructuralAnalyzer, inputs: dict[str, float], output: str) -> object:
    request = EvaluationRequest(output=output, input_domains=dict.fromkeys(inputs, Domain.scalar()))
    with EvaluationContext():
        case = analyser.analyse(request)
        compiled = analyser.backend.chain_case(case)
        return compiled.callable(**{mangle_name(name): value for name, value in inputs.items()})


@pytest.mark.parametrize("child_order", ((0, 1), (1, 0)))
@pytest.mark.parametrize("parent_order", tuple(itertools.permutations(range(3))))
def test_multiple_child_outputs_execute_in_every_declaration_order(
    child_order: tuple[int, ...], parent_order: tuple[int, ...]
) -> None:
    operations = (b, next_value)
    child = SubSystemConstituent("child", Implementation(Ports, [operations[index] for index in child_order]))
    constituents = (child, supply, combined)
    analyser = CausalizationStructuralAnalyzer([constituents[index] for index in parent_order], PYTHON_BACKEND)

    expected = 13.0
    assert _run(analyser, {"source": 3.0}, "answer") == expected


@pytest.mark.parametrize("reverse", (False, True))
def test_dependency_settlement_is_independent_of_assignment_insertion_order(*, reverse: bool) -> None:
    child = SubSystemConstituent("child", Implementation(Ports, [b, next_value]))
    outputs = ["child.b", "child.c"]
    if reverse:
        outputs.reverse()
    steps, rejected = settle_assignments(dict.fromkeys(outputs, child), {"child.a"}, PYTHON_BACKEND)
    assert rejected == frozenset()
    assert {step.output: step.child_inputs for step in steps} == {
        "child.b": frozenset({"a"}),
        "child.c": frozenset({"b"}),
    }


def test_direct_cycle_reports_unsupported_blocks() -> None:
    @transform
    def x(y: float) -> float:
        return y + 1

    @transform
    def y(x: float) -> float:
        return x / 2

    analyser = CausalizationStructuralAnalyzer([x, y], PYTHON_BACKEND)
    with pytest.raises(BiltExecutionPlanError, match="coupled blocks"):
        _run(analyser, {}, "x")


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
    steps, rejected = settle_assignments({name: operations[name] for name in order}, {"source"}, PYTHON_BACKEND)
    graph, unresolved = build_dependency_graph(steps, {"source"})
    assert rejected == frozenset()
    assert unresolved == {}
    assert rx.is_directed_acyclic_graph(graph)
    assert tuple(step.output for step in steps) == ("a", "b", "z")


def test_stalled_primitive_keeps_missing_dependency() -> None:
    @transform
    def result(missing: float) -> float:
        return missing + 1

    steps, rejected = settle_assignments({"result": result}, set(), PYTHON_BACKEND)
    _, unresolved = build_dependency_graph(steps, set())
    assert rejected == frozenset()
    assert unresolved == {"result": frozenset({"missing"})}


def test_structural_underconstraint_keeps_its_error_type() -> None:
    @equation
    def relation(x: float, y: float) -> bool:
        return x == y + 1

    analyser = CausalizationStructuralAnalyzer([relation], PYTHON_BACKEND)
    with pytest.raises(BiltUnderconstrainedError):
        _run(analyser, {}, "x")


@pytest.mark.parametrize("reverse", (False, True))
def test_later_parent_producer_unlocks_dependent_child_routes(*, reverse: bool) -> None:
    @transform
    def z(source: float) -> float:
        return source * 2

    @equation
    def feed(z: float, child: Ports) -> bool:
        return child.a == z

    child = SubSystemConstituent("child", Implementation(Ports, [b, next_value]))
    assignments = {"child.a": feed, "child.b": child, "child.c": child, "z": z}
    if reverse:
        assignments = dict(reversed(tuple(assignments.items())))
    steps, rejected = settle_assignments(assignments, {"source"}, PYTHON_BACKEND)
    by_output = {step.output: step for step in steps}
    assert rejected == frozenset()
    assert by_output["child.b"].child_inputs == frozenset({"a"})
    assert by_output["child.c"].child_inputs == frozenset({"b"})
    graph, unresolved = build_dependency_graph(steps, {"source"})
    assert unresolved == {}
    assert rx.is_directed_acyclic_graph(graph)
