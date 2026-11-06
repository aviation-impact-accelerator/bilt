import itertools
import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import expand_structure
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import StructuralAnalysis
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_request_structure
from bilt.core._sub_system import SubSystemConstituent
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._structural_analysis._causalization_block_solver._structural_scope import StructuralScope


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()
    x: float = field()
    y: float = field()


@equation
def relation(a: float, b: float) -> bool:
    return b == 2 * a


@equation
def supply(source: float, child: Ports) -> bool:
    return child.a == source


@equation
def result(answer: float, child: Ports) -> bool:
    return answer == child.b


def test_expansion_discovers_parent_connections_to_child_inputs() -> None:
    child = SubSystemConstituent("child", Implementation(Ports, [relation]))
    incidence = FullIncidenceGraph([child, supply, result])
    validate_request_structure(incidence, {"source"}, "answer")
    graph, attributes = expand_structure(incidence, {"source"}, "answer")

    assert set(attributes) == {"answer", "child.a", "child.b"}
    assert {graph[node] for node in graph.node_indices() if node not in attributes.values()} == {
        "child.relation",
        "supply",
        "result",
    }


@pytest.mark.parametrize("equivalent", (False, True))
def test_parent_matching_cannot_hide_child_competing_constraint(*, equivalent: bool) -> None:
    @equation
    def parent_answer(answer: float, source: float) -> bool:
        return answer == 2 * source

    @equation
    def contradictory(a: float, b: float) -> bool:
        return b == 3 * a

    child = SubSystemConstituent("child", Implementation(Ports, [relation if equivalent else contradictory]))
    incidence = FullIncidenceGraph([parent_answer, supply, result, child])
    with pytest.raises(BiltOverconstrainedError, match="child"):
        validate_request_structure(incidence, {"source"}, "answer")


def test_competing_equations_inside_a_child_are_all_visible() -> None:
    @equation
    def other(a: float, b: float) -> bool:
        return b == a + a

    child = SubSystemConstituent("child", Implementation(Ports, [relation, other]))
    with pytest.raises(BiltOverconstrainedError, match=r"child\.other"):
        validate_request_structure(FullIncidenceGraph([child]), {"child.a"}, "child.b")
    with pytest.raises(BiltOverconstrainedError):
        validate_request_structure(FullIncidenceGraph([relation, other]), {"a"}, "b")


def test_coupled_child_is_structurally_valid_without_an_executable_route(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def total(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def difference(b: float, x: float, y: float) -> bool:
        return x - y == b

    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("Structural validation must not select or realise child routes.")

    monkeypatch.setattr(Implementation, "analyser_for", forbidden)
    monkeypatch.setattr(SubSystemConstituent, "required_inputs_for", forbidden)
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", forbidden)
    child = SubSystemConstituent("child", Implementation(Ports, [total, difference]))
    incidence = FullIncidenceGraph([child])
    validate_request_structure(incidence, {"child.a", "child.b"}, "child.x")
    graph, attributes = expand_structure(incidence, {"child.a", "child.b"}, "child.x")
    expected_edges = 4
    assert set(attributes) == {"child.x", "child.y"}
    assert graph.num_edges() == expected_edges


def test_nested_names_are_qualified_and_parent_scope_is_revisited() -> None:
    inner = SubSystemConstituent("inner", Implementation(Ports, [relation]))

    @equation
    def inner_supply(a: float, inner: Ports) -> bool:
        return inner.a == a

    @equation
    def inner_result(b: float, inner: Ports) -> bool:
        return b == inner.b

    child = SubSystemConstituent("child", Implementation(Ports, [inner, inner_supply, inner_result]))
    incidence = FullIncidenceGraph([child, supply, result])
    validate_request_structure(incidence, {"source"}, "answer")
    graph, attributes = expand_structure(incidence, {"source"}, "answer")
    assert "child.inner.a" in attributes
    assert "child.inner.relation" in graph.nodes()
    assert "supply" in graph.nodes()


def test_shared_implementation_preserves_each_instance_constraint() -> None:
    implementation = Implementation(Ports, [relation])
    left = SubSystemConstituent("left", implementation)
    right = SubSystemConstituent("right", implementation)

    @equation
    def combined(answer: float, left: Ports, right: Ports) -> bool:
        return answer == left.b + right.b

    incidence = FullIncidenceGraph([left, right, combined])
    validate_request_structure(incidence, {"left.a", "right.a"}, "answer")
    graph, _ = expand_structure(incidence, {"left.a", "right.a"}, "answer")
    assert "left.relation" in graph.nodes()
    assert "right.relation" in graph.nodes()


def test_unused_child_sharing_a_supplied_input_stays_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    child = SubSystemConstituent("child", Implementation(Ports, [relation]))

    @transform
    def answer(source: float) -> float:
        return 2 * source

    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("The unused child must remain unopened.")

    monkeypatch.setattr(Implementation, "structural_analysis", forbidden)
    validate_request_structure(FullIncidenceGraph([answer, child, supply]), {"source"}, "answer")


def test_child_transform_is_not_used_backwards() -> None:
    @transform
    def b(a: float) -> float:
        return 2 * a

    child = SubSystemConstituent("child", Implementation(Ports, [b]))
    with pytest.raises(BiltUnderconstrainedError, match=r"child\.a"):
        validate_request_structure(FullIncidenceGraph([child]), {"child.b"}, "child.a")


def test_missing_child_route_reports_the_request() -> None:
    child = SubSystemConstituent("child", Implementation(Ports, []))
    with pytest.raises(BiltUnderconstrainedError, match="Cannot compute answer"):
        validate_request_structure(FullIncidenceGraph([child, result]), set(), "answer")


def test_supplied_target_needs_no_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    child = SubSystemConstituent("child", Implementation(Ports, [relation]))

    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("A supplied target needs no child expansion.")

    monkeypatch.setattr(Implementation, "structural_analysis", forbidden)
    validate_request_structure(FullIncidenceGraph([child]), {"child.b"}, "child.b")


@pytest.mark.parametrize("child_order", ((0, 1), (1, 0)))
@pytest.mark.parametrize("parent_order", tuple(itertools.permutations(range(3))))
def test_dependent_child_outputs_are_valid_in_every_declaration_order(
    child_order: tuple[int, ...], parent_order: tuple[int, ...]
) -> None:
    @transform
    def b(a: float) -> float:
        return 2 * a

    @equation
    def next_value(b: float, c: float) -> bool:
        return c == b + 1

    @equation
    def combined(answer: float, child: Ports) -> bool:
        return answer == child.b + child.c

    operations = (b, next_value)
    child = SubSystemConstituent("child", Implementation(Ports, [operations[index] for index in child_order]))
    constituents = (child, supply, combined)
    incidence = FullIncidenceGraph([constituents[index] for index in parent_order])
    validate_request_structure(incidence, {"source"}, "answer")


def test_validation_is_unchanged_after_child_realization_failure() -> None:
    @equation
    def square(a: float, b: float) -> bool:
        return a == b * b

    child = SubSystemConstituent("child", Implementation(Ports, [square]))
    incidence = FullIncidenceGraph([child])
    validate_request_structure(incidence, {"child.a"}, "child.b")
    analyser = CausalizationStructuralAnalyzer([child], PYTHON_BACKEND)
    request = EvaluationRequest(output="child.b", input_domains={"child.a": Domain.scalar()})
    with EvaluationContext(), pytest.raises(BiltAnalysisError):
        analyser.analyse(request)
    validate_request_structure(incidence, {"child.a"}, "child.b")


def test_each_child_attribute_is_queried_once_per_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def next_value(b: float, c: float) -> bool:
        return c == b + 1

    implementation = Implementation(Ports, [relation, next_value])
    child = SubSystemConstituent("child", implementation)
    original = StructuralAnalysis.inspect
    requests: list[frozenset[str]] = []

    def inspect(
        self: StructuralAnalysis, available: collections.abc.Set[str], required: collections.abc.Set[str]
    ) -> StructuralScope:
        requests.append(frozenset(required))
        return original(self, available, required)

    monkeypatch.setattr(StructuralAnalysis, "inspect", inspect)
    validate_request_structure(FullIncidenceGraph([child]), {"child.a"}, "child.c")
    assert len(requests) == len(set(requests))
    assert {"b", "c"} <= set().union(*requests)


def test_recursive_containment_is_rejected_without_unbounded_expansion() -> None:
    loop = SubSystemConstituent("loop", Implementation(Ports, []))

    @equation
    def connect(a: float, loop: Ports) -> bool:
        return a == loop.a

    implementation = Implementation(Ports, [loop, connect])
    loop.implementation = implementation
    with pytest.raises(BiltAnalysisError, match="Recursive subsystem containment"):
        validate_request_structure(FullIncidenceGraph([loop, connect]), set(), "a")


def test_feedback_across_a_child_boundary_is_retained_structurally() -> None:
    child = SubSystemConstituent("child", Implementation(Ports, [relation]))

    @equation
    def feedback(child: Ports) -> bool:
        return child.a == child.b / 4

    incidence = FullIncidenceGraph([child, feedback, result])
    validate_request_structure(incidence, set(), "answer")
    graph, attributes = expand_structure(incidence, set(), "answer")
    assert {"child.a", "child.b"} <= attributes.keys()
    assert "child.relation" in graph.nodes()
    assert "feedback" in graph.nodes()


def test_feedback_between_sibling_children_is_retained_structurally() -> None:
    implementation = Implementation(Ports, [relation])
    left = SubSystemConstituent("left", implementation)
    right = SubSystemConstituent("right", implementation)

    @equation
    def forward(left: Ports, right: Ports) -> bool:
        return right.a == left.b / 4

    @equation
    def feedback(left: Ports, right: Ports) -> bool:
        return left.a == right.b / 4

    @equation
    def answer_eq(answer: float, left: Ports) -> bool:
        return answer == left.b

    validate_request_structure(FullIncidenceGraph([left, right, forward, feedback, answer_eq]), set(), "answer")
