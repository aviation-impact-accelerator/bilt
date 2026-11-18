"""Check incremental structural discovery against full request-local inspection."""

import itertools

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._structural_analysis._causalization_block_solver import _structural_scope as scope_module
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_scope import (
    StructuralScope,
    _IncrementalStructuralScope,
    inspect_structure,
)
from bilt.core._sub_system import SubSystemConstituent


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@equation
def first(a: float, b: float) -> bool:
    return b == a + 1


@equation
def second(b: float, c: float) -> bool:
    return c == b * 2


@equation
def together(a: float, b: float, c: float) -> bool:
    return c == a + b


@transform
def c(b: float) -> float:
    return 3 * b


@transform
def b(a: float) -> float:
    return a + 2


def _scope_signature(scope: object) -> tuple[object, ...]:
    assert isinstance(scope, StructuralScope)
    return (
        frozenset(id(operation) for operation in scope.constraints),
        frozenset(
            (id(boundary.operation), boundary.available, boundary.required) for boundary in scope.boundaries
        ),
        scope.attributes,
    )


def _assert_incremental_matches_full(
    incidence: FullIncidenceGraph,
    supplied: set[str],
    requests: tuple[frozenset[str], ...],
) -> None:
    scope = _IncrementalStructuralScope(incidence)
    accumulated_constraints: set[int] = set()
    accumulated_ports: dict[int, set[str]] = {}
    attributes: set[str] = set()
    targets: set[str] = set()
    for required in requests:
        delta = scope.inspect(supplied, required)
        targets.update(required)
        attributes.update(delta.attributes)
        accumulated_constraints.update(id(operation) for operation in delta.constraints)
        for boundary in delta.boundaries:
            accumulated_ports.setdefault(id(boundary.operation), set()).update(boundary.required)

        reference = inspect_structure(incidence, supplied, targets)
        actual_ports = frozenset((identity, frozenset(ports)) for identity, ports in accumulated_ports.items())
        expected_ports = frozenset(
            (id(boundary.operation), boundary.required) for boundary in reference.boundaries
        )
        assert accumulated_constraints == {id(operation) for operation in reference.constraints}
        assert actual_ports == expected_ports
        assert attributes == reference.attributes

        # Repeated requirements are a no-op, including repeated boundary ports.
        repeated = scope.inspect(supplied, required)
        assert repeated.constraints == ()
        assert repeated.boundaries == ()
        assert repeated.attributes == required


def test_incremental_scope_matches_all_mixed_constituent_subsets() -> None:
    operations = (first, second, together, b, c)
    for count in range(1, len(operations) + 1):
        for selected in itertools.combinations(operations, count):
            incidence = FullIncidenceGraph(selected)
            for supplied in (set(), {"a"}, {"b"}, {"c"}, {"a", "b"}, {"a", "c"}, {"b", "c"}):
                _assert_incremental_matches_full(
                    incidence, supplied, (frozenset({"a"}), frozenset({"c"}), frozenset({"b"}))
                )


@pytest.mark.parametrize("reverse", (False, True))
def test_incremental_boundaries_preserve_instances_and_ports(reverse: bool) -> None:
    shared = Implementation(Ports, [first, second])
    left = SubSystemConstituent("left", shared)
    right = SubSystemConstituent("right", shared)

    @equation
    def output(result: float, left: Ports, right: Ports) -> bool:
        return result == left.c + right.c

    operations = [left, right, output]
    if reverse:
        operations.reverse()
    incidence = FullIncidenceGraph(operations)
    _assert_incremental_matches_full(
        incidence,
        set(),
        (frozenset({"result"}), frozenset({"left.a"}), frozenset({"right.b"})),
    )


def test_incremental_scope_restarts_for_different_supplied_inputs() -> None:
    incidence = FullIncidenceGraph([first, second, b])
    scope = _IncrementalStructuralScope(incidence)
    for available in (set(), {"a"}, {"b"}, {"a", "b"}, set()):
        delta = scope.inspect(available, {"c"})
        assert _scope_signature(delta) == _scope_signature(inspect_structure(incidence, available, {"c"}))


def test_full_nested_expansion_preserves_independent_scope_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = Implementation(Ports, [first, second])
    child = SubSystemConstituent("child", shared)

    @equation
    def output(result: float, child: Ports) -> bool:
        return result == child.c

    graph = FullIncidenceGraph([child, output])
    with monkeypatch.context() as patched:
        patched.setattr(
            scope_module._IncrementalStructuralScope,  # noqa: SLF001
            "inspect",
            lambda self, available, required: inspect_structure(self.incidence, available, required),
        )
        original = StructuralExpansion(graph, {"child.a"}, "result")
        original.collect()
        expected = original.snapshot()

    actual = StructuralExpansion(graph, {"child.a"}, "result")
    actual.collect()
    snapshot = actual.snapshot()
    assert snapshot.attributes == expected.attributes
    assert snapshot.available == expected.available
    assert snapshot.expanded_paths == expected.expanded_paths
    assert frozenset((row.path, id(row.operation), row.attributes) for row in snapshot.constraints) == frozenset(
        (row.path, id(row.operation), row.attributes) for row in expected.constraints
    )
    assert snapshot.complete
    assert expected.complete
