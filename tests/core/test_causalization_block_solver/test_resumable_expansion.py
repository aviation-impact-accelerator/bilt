import typing

import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.core._errors import BiltAnalysisError
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltOverconstrainedError
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_structure
from bilt.core._sub_system import SubSystemConstituent


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()


@equation
def relation(a: float, b: float) -> bool:
    return b == 2 * a


def _child(name: str = "child") -> SubSystemConstituent:
    return SubSystemConstituent(name, Implementation(Ports, [relation]))


def test_initial_scope_stays_opaque_and_cannot_be_finalized(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: object, **__: object) -> typing.Never:
        pytest.fail("Constructing a partial expansion must not open a child.")

    child = _child()
    monkeypatch.setattr(Implementation, "structural_analysis", forbidden)
    expansion = StructuralExpansion(FullIncidenceGraph([child]), {"child.a"}, "child.b")
    snapshot = expansion.snapshot()
    assert not snapshot.complete
    assert snapshot.constraints == ()
    assert snapshot.unresolved[0].path == ("child",)
    assert snapshot.unresolved[0].boundary.available == frozenset({"a"})
    assert snapshot.unresolved[0].boundary.required == frozenset({"b"})
    with pytest.raises(BiltAnalysisError, match="unresolved"):
        snapshot.to_incidence()


def test_expansion_preserves_source_identity_and_prior_snapshot() -> None:
    expansion = StructuralExpansion(FullIncidenceGraph([_child()]), {"child.a"}, "child.b")
    before = expansion.snapshot()
    after = expansion.expand(("child",))
    assert after.complete
    assert not before.complete
    assert before.constraints == ()
    assert after.constraints[0].operation is relation
    assert after.constraints[0].path == ("child",)
    assert after.constraints[0].attributes == frozenset({"child.a", "child.b"})
    assert after.expanded_paths == frozenset({("child",)})
    assert after.query_count > before.query_count
    assert expansion.expand(("child",)) == after
    with pytest.raises(KeyError):
        expansion.expand(("unknown",))


def test_nested_child_remains_unresolved_after_opening_its_parent() -> None:
    inner = _child("inner")

    @equation
    def connection(b: float, inner: Ports) -> bool:
        return b == inner.b

    outer = SubSystemConstituent("outer", Implementation(Ports, [inner, connection]))
    expansion = StructuralExpansion(FullIncidenceGraph([outer]), {"outer.inner.a"}, "outer.b")
    first = expansion.expand(("outer",))
    assert [item.path for item in first.unresolved] == [("outer", "inner")]
    assert [row.operation for row in first.constraints] == [connection]
    assert expansion.expand(("outer", "inner")).complete


@pytest.mark.parametrize("reverse", (False, True))
def test_shared_children_can_be_expanded_in_either_order(*, reverse: bool) -> None:
    implementation = Implementation(Ports, [relation])
    left = SubSystemConstituent("left", implementation)
    right = SubSystemConstituent("right", implementation)

    @equation
    def combined(answer: float, left: Ports, right: Ports) -> bool:
        return answer == left.b + right.b

    expansion = StructuralExpansion(FullIncidenceGraph([left, right, combined]), {"left.a", "right.a"}, "answer")
    paths = [("left",), ("right",)]
    if reverse:
        paths.reverse()
    first = expansion.expand(paths[0])
    assert [item.path for item in first.unresolved] == [paths[1]]
    final = expansion.expand(paths[1])
    rows = [row for row in final.constraints if row.operation is relation]
    assert {row.path for row in rows} == {("left",), ("right",)}
    graph, attributes = final.to_incidence()
    partition = validate_structure(graph, frozenset(attributes.values()), frozenset())
    assert not partition.underdetermined
    assert not partition.overdetermined


def test_child_discovery_revisits_parent_connections_and_merges_ports() -> None:
    @equation
    def supply(source: float, child: Ports) -> bool:
        return child.a == source

    @equation
    def conflict(child: Ports) -> bool:
        return child.a == child.b / 3

    expansion = StructuralExpansion(FullIncidenceGraph([_child(), supply, conflict]), {"source"}, "child.b")
    before = expansion.snapshot()
    assert before.unresolved[0].boundary.required == frozenset({"a", "b"})
    final = expansion.expand(("child",))
    assert {row.operation for row in final.constraints} == {supply, conflict, relation}
    graph, attributes = final.to_incidence()
    with pytest.raises(BiltOverconstrainedError):
        validate_structure(graph, frozenset(attributes.values()), frozenset())


def test_exposing_a_child_input_discovers_a_previously_hidden_parent_equation() -> None:
    @equation
    def supply(source: float, child: Ports) -> bool:
        return child.a == source

    expansion = StructuralExpansion(FullIncidenceGraph([_child(), supply]), {"source"}, "child.b")
    assert expansion.snapshot().constraints == ()
    final = expansion.expand(("child",))
    assert {row.operation for row in final.constraints} == {relation, supply}
    assert len(final.constraints) == len({(row.path, id(row.operation)) for row in final.constraints})


def test_supplied_target_needs_no_child_expansion() -> None:
    expansion = StructuralExpansion(FullIncidenceGraph([_child()]), {"child.b"}, "child.b")
    snapshot = expansion.snapshot()
    assert snapshot.complete
    assert snapshot.constraints == ()
    graph, attributes = snapshot.to_incidence()
    assert graph.num_nodes() == 0
    assert attributes == {}
