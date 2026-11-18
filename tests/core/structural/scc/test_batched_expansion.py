"""Compare batched structural inspection with the original per-attribute traversal."""

import itertools

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_structure
from bilt.core._sub_system import SubSystemConstituent


@system
class Ports(metaclass=Interface):
    a: float = field()
    b: float = field()
    c: float = field()


@equation
def first(a: float, b: float) -> bool:
    return b == 2 * a


@equation
def second(b: float, c: float) -> bool:
    return c == b + 1


@equation
def combined_inputs(a: float, b: float, c: float) -> bool:
    return c == a + b


@transform
def b(a: float) -> float:
    return 2 * a


@transform
def c(b: float) -> float:
    return b + 1


@equation
def conflict(a: float, b: float) -> bool:
    return b == 3 * a


@equation
def supply(source: float, child: Ports) -> bool:
    return child.a == source


@equation
def result(answer: float, child: Ports) -> bool:
    return answer == child.c


class _UnbatchedExpansion(StructuralExpansion):
    def _drain(self) -> None:
        while self.pending:
            namespace, attribute = self.pending.pop()
            prefix = ".".join((*namespace, "")) if namespace else ""
            supplied = {name[len(prefix) :] for name in self.available if name.startswith(prefix)}
            scope = self.contexts[namespace].query(supplied, {attribute})
            self.query_count += 1
            self._merge(namespace, scope)


def _structure(expansion: StructuralExpansion) -> tuple[object, ...]:
    snapshot = expansion.snapshot()
    rows = frozenset((row.path, id(row.operation), row.attributes) for row in snapshot.constraints)
    boundaries = frozenset(
        (
            unresolved.path,
            id(unresolved.boundary.operation),
            unresolved.boundary.available,
            unresolved.boundary.required,
        )
        for unresolved in snapshot.unresolved
    )
    return rows, boundaries, snapshot.attributes, snapshot.available, snapshot.expanded_paths, snapshot.complete


def _validation(expansion: StructuralExpansion, output: str) -> tuple[object, ...]:
    graph, attributes = expansion.collect()
    partition = None
    error = None
    try:
        partition = validate_structure(
            graph,
            frozenset(attributes.values()),
            frozenset(),
            required_nodes=frozenset({attributes[output]}),
        )
    except (BiltOverconstrainedError, BiltUnderconstrainedError) as caught:
        error = type(caught), str(caught)
    if partition is not None:
        return (
            frozenset(attributes),
            frozenset(graph.nodes()),
            frozenset((graph[first], graph[second]) for first, second in graph.edge_list()),
            frozenset(graph[node] for node in partition.well_determined),
            frozenset(graph[node] for node in partition.underdetermined),
            frozenset(graph[node] for node in partition.overdetermined),
        )
    return (frozenset(attributes), error)


def _compare(incidence: FullIncidenceGraph, available: set[str], output: str) -> tuple[int, int]:
    original = _UnbatchedExpansion(incidence, available, output)
    batched = StructuralExpansion(incidence, available, output)
    assert _structure(batched) == _structure(original)
    while original.snapshot().unresolved:
        path = original.snapshot().unresolved[0].path
        assert path in {entry.path for entry in batched.snapshot().unresolved}
        batched.expand(path)
        original.expand(path)
        assert _structure(batched) == _structure(original)
    assert _validation(batched, output) == _validation(original, output)
    return original.query_count, batched.query_count


def test_batches_multiple_pending_attributes_without_losing_constraints() -> None:
    original, batched = _compare(FullIncidenceGraph([combined_inputs]), set(), "c")
    assert batched < original


@pytest.mark.parametrize("reverse", (False, True))
def test_shared_children_and_nested_dependencies_match_unbatched(reverse: bool) -> None:
    shared = Implementation(Ports, [first, second])
    left = SubSystemConstituent("left", shared)
    right = SubSystemConstituent("right", shared)

    @equation
    def combined(answer: float, left: Ports, right: Ports) -> bool:
        return answer == left.c + right.c

    ops = [left, right, combined]
    if reverse:
        ops.reverse()
    old, new = _compare(FullIncidenceGraph(ops), {"left.a", "right.a"}, "answer")
    assert new <= old


def test_parent_connections_and_nested_child_boundaries_match_unbatched() -> None:
    inner = SubSystemConstituent("inner", Implementation(Ports, [first, second]))

    @equation
    def pass_through(a: float, inner: Ports) -> bool:
        return inner.a == a

    @equation
    def pass_result(c: float, inner: Ports) -> bool:
        return c == inner.c

    child = SubSystemConstituent("child", Implementation(Ports, [inner, pass_through, pass_result]))
    original, batched = _compare(FullIncidenceGraph([child, supply, result]), {"source"}, "answer")
    assert batched <= original


@pytest.mark.parametrize("overconstrained", (False, True))
def test_feedback_and_competing_constraints_match_unbatched(overconstrained: bool) -> None:
    child = SubSystemConstituent(
        "child",
        Implementation(Ports, [first, second, *([conflict] if overconstrained else [])]),
    )

    @equation
    def feedback(child: Ports) -> bool:
        return child.a == child.b / 4

    _compare(FullIncidenceGraph([child, feedback, result]), set(), "answer")


def test_underconstrained_child_matches_unbatched() -> None:
    child = SubSystemConstituent("child", Implementation(Ports, [second]))
    _compare(FullIncidenceGraph([child, result]), set(), "answer")


def test_small_mixed_equation_and_transform_graphs_match_unbatched() -> None:
    operations = (first, second, combined_inputs, b, c)
    checked = 0
    for count in range(1, len(operations) + 1):
        for selected in itertools.combinations(operations, count):
            for supplied in (
                set(),
                {"a"},
                {"b"},
                {"c"},
                {"a", "b"},
                {"a", "c"},
                {"b", "c"},
                {"a", "b", "c"},
            ):
                for output in ("a", "b", "c"):
                    if output in supplied:
                        continue
                    _compare(FullIncidenceGraph(selected), supplied, output)
                    checked += 1
    expected_cases = 372
    assert checked == expected_cases


@pytest.mark.parametrize("reverse", (False, True))
def test_transform_inputs_remain_directional_across_child_boundaries(reverse: bool) -> None:
    child_ops = [b, c]
    if reverse:
        child_ops.reverse()
    child = SubSystemConstituent("child", Implementation(Ports, child_ops))
    _compare(FullIncidenceGraph([child]), {"child.a"}, "child.c")
    _compare(FullIncidenceGraph([child]), {"child.c"}, "child.a")
    _compare(FullIncidenceGraph([child]), {"child.c"}, "child.b")
