import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._errors import BiltAnalysisError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver._backend import CausalizationStructuralAnalyzer
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import (
    StructuralAnalysis,
    inspect_structure,
)
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    from bilt.core._model_constituent import BiltModelConstituent


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@equation
def relation(a: float, b: float) -> bool:
    return b == 2 * a


@transform
def c(b: float) -> float:
    return b + 1


@pytest.mark.parametrize("reverse", (False, True))
def test_scope_retains_all_competing_constraints(*, reverse: bool) -> None:

    @equation
    def other(a: float, b: float) -> bool:
        return b == a + a

    operations = [relation, other]
    if reverse:
        operations.reverse()
    scope = Implementation(Ports, operations).structural_analysis().inspect({"a"}, {"b"})

    assert set(scope.constraints) == {relation, other}
    assert scope.attributes == frozenset({"a", "b"})
    assert scope.boundaries == ()


def test_scope_retains_coupled_equations_without_isolation() -> None:

    @equation
    def total(a: float, x: float, y: float) -> bool:
        return x + y == a

    @equation
    def difference(b: float, x: float, y: float) -> bool:
        return x - y == b

    scope = Implementation(Ports, [total, difference]).structural_analysis().inspect({"a", "b"}, {"x"})

    assert set(scope.constraints) == {total, difference}
    assert scope.attributes == frozenset({"a", "b", "x", "y"})
    assert scope.required == frozenset({"x"})


def test_scope_stops_at_supplied_values() -> None:
    implementation = Implementation(Ports, [relation, c])
    scope = implementation.structural_analysis().inspect({"b"}, {"c"})

    assert scope.constraints == (c,)
    assert scope.attributes == frozenset({"b", "c"})


def test_supplied_target_needs_no_constraints() -> None:
    scope = Implementation(Ports, [relation]).structural_analysis().inspect({"b"}, {"b"})

    assert scope.constraints == ()
    assert scope.boundaries == ()
    assert scope.required == scope.available == frozenset({"b"})


def test_unknown_target_remains_visible_without_classification() -> None:
    scope = Implementation(Ports, [relation]).structural_analysis().inspect({"a"}, {"missing"})

    assert scope.constraints == ()
    assert scope.attributes == scope.required == frozenset({"missing"})


def test_transform_is_not_traversed_backwards() -> None:
    scope = Implementation(Ports, [c]).structural_analysis().inspect({"c"}, {"b"})

    assert scope.constraints == ()
    assert scope.attributes == frozenset({"b"})


def test_duplicate_discovery_preserves_source_identity_once() -> None:
    incidence = FullIncidenceGraph([relation, relation])
    before = list(incidence.graph.edge_list())
    scope = inspect_structure(incidence, set(), {"a", "b"})

    assert scope.constraints == (relation,)
    assert scope.constraints[0] is relation
    assert incidence.graph.edge_list() == before


def test_child_boundary_is_lazy_and_uses_local_port_names(monkeypatch: pytest.MonkeyPatch) -> None:
    child_implementation = Implementation(Ports, [relation])
    child = SubSystemConstituent("child", child_implementation)
    parent = Implementation(Ports, [child])
    original = FullIncidenceGraph.__init__
    inspected: list[tuple[BiltModelConstituent, ...]] = []

    def initialise(self: FullIncidenceGraph, constituents: collections.abc.Iterable[BiltModelConstituent]) -> None:
        operations = tuple(constituents)
        inspected.append(operations)
        original(self, operations)

    monkeypatch.setattr(FullIncidenceGraph, "__init__", initialise)
    scope = parent.structural_analysis().inspect({"child.a"}, {"child.b"})
    assert inspected == [(child,)]
    assert scope.constraints == ()
    assert len(scope.boundaries) == 1
    boundary = scope.boundaries[0]
    assert boundary.operation is child
    assert boundary.required == frozenset({"b"})
    assert boundary.available == frozenset({"a"})

    expanded = StructuralAnalysis.inspect_boundary(boundary)
    assert inspected == [(child,), (relation,)]
    assert expanded.constraints == (relation,)
    assert StructuralAnalysis.inspect_boundary(boundary) == expanded
    assert inspected == [(child,), (relation,)]


def test_nested_boundaries_remain_unresolved_until_queried() -> None:
    inner = SubSystemConstituent("inner", Implementation(Ports, [relation]))

    @equation
    def result(b: float, inner: Ports) -> bool:
        return b == inner.b

    outer = SubSystemConstituent("outer", Implementation(Ports, [inner, result]))
    parent = Implementation(Ports, [outer])
    root = parent.structural_analysis().inspect(set(), {"outer.b"})
    child = StructuralAnalysis.inspect_boundary(root.boundaries[0])

    assert root.constraints == ()
    assert child.constraints == (result,)
    assert child.boundaries[0].operation is inner
    assert StructuralAnalysis.inspect_boundary(child.boundaries[0]).constraints == (relation,)


def test_shared_children_have_distinct_boundaries_and_shared_sources() -> None:
    implementation = Implementation(Ports, [relation])
    left = SubSystemConstituent("left", implementation)
    right = SubSystemConstituent("right", implementation)
    parent = Implementation(Ports, [left, right])
    scope = parent.structural_analysis().inspect({"left.a", "right.a"}, {"left.b", "right.b"})

    assert {boundary.operation for boundary in scope.boundaries} == {left, right}
    assert all(
        StructuralAnalysis.inspect_boundary(boundary).constraints[0] is relation for boundary in scope.boundaries
    )
    left_only = parent.structural_analysis().inspect({"left.a"}, {"left.b"})
    assert tuple(boundary.operation for boundary in left_only.boundaries) == (left,)


def test_inspection_does_not_invoke_planning_or_realization(monkeypatch: pytest.MonkeyPatch) -> None:

    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("Structural inspection must not invoke execution planning or realization.")

    monkeypatch.setattr(Implementation, "analyser_for", forbidden)
    monkeypatch.setattr(Implementation, "get_evaluation", forbidden)
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", forbidden)
    monkeypatch.setattr(SubSystemConstituent, "required_inputs_for", forbidden)
    child = SubSystemConstituent("child", Implementation(Ports, [relation]))
    parent = Implementation(Ports, [child])

    scope = parent.structural_analysis().inspect({"child.a"}, {"child.b"})
    assert StructuralAnalysis.inspect_boundary(scope.boundaries[0]).constraints == (relation,)


@pytest.mark.parametrize("analyser_type", (ExhaustiveStructuralAnalyser, CausalizationStructuralAnalyzer))
def test_inspection_is_unchanged_after_failed_isolation(analyser_type: type, monkeypatch: pytest.MonkeyPatch) -> None:

    @equation
    def square(a: float, b: float) -> bool:
        return a == b * b

    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", analyser_type)
    implementation = Implementation(Ports, [square])
    before = implementation.structural_analysis().inspect({"a"}, {"b"})
    analyser = implementation.analyser_for(PYTHON_BACKEND)
    request = EvaluationRequest(output="b", input_domains={"a": Domain.scalar()})
    with EvaluationContext(), pytest.raises(BiltAnalysisError):
        analyser.analyse(request)

    assert implementation.structural_analysis().inspect({"a"}, {"b"}) == before


def test_analysis_owns_the_constituents_passed_at_construction() -> None:
    components = [relation, c]
    analysis = StructuralAnalysis(components)
    components.clear()

    assert analysis.inspect({"a"}, {"c"}).constraints == (relation, c)


def test_implementation_reuses_its_structural_analysis_object() -> None:
    implementation = Implementation(Ports, [relation])
    assert implementation.structural_analysis() is implementation.structural_analysis()
