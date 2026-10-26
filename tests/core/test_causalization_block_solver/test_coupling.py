import dataclasses
import itertools
import typing

import pytest
import rustworkx as rx

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._coupling import unavoidable_coupled_outputs
from bilt.core._structural_analysis._causalization_block_solver._errors import BiltExecutionPlanError
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import (
    ExpandedConstraint,
    ExpansionSnapshot,
    StructuralExpansion,
)
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_request_structure
from bilt.core._sub_system import SubSystemConstituent
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.core._model_constituent import BiltModelConstituent


@system
class Ports(metaclass=Interface):
    a: float = typing.cast("float", field())
    b: float = typing.cast("float", field())
    x: float = typing.cast("float", field())
    y: float = typing.cast("float", field())


@equation
def total(x: float, y: float, a: float) -> bool:
    return x + y == a


@equation
def difference(x: float, y: float, b: float) -> bool:
    return x - y == b


def _nested(depth: int) -> Implementation:
    implementation = Implementation(Ports, [total, difference])
    for _ in range(depth):
        inner = SubSystemConstituent("inner", implementation)

        @equation
        def input_a(a: float, inner: Ports) -> bool:
            return a == inner.a

        @equation
        def input_b(b: float, inner: Ports) -> bool:
            return b == inner.b

        @equation
        def output_x(x: float, inner: Ports) -> bool:
            return x == inner.x

        implementation = Implementation(Ports, [inner, input_a, input_b, output_x])
    return implementation


def _unexpected_search(*_: object, **__: object) -> typing.Never:
    message = "Unavoidable coupled requests must be rejected before execution search."
    raise AssertionError(message)


@pytest.mark.parametrize("depth", (1, 2))
def test_rejects_nested_coupling_before_execution_search(monkeypatch: pytest.MonkeyPatch, depth: int) -> None:
    implementation = _nested(depth)
    monkeypatch.setattr(search, "find_matching", _unexpected_search)
    monkeypatch.setattr(SubSystemConstituent, "required_inputs_for", _unexpected_search)
    # Construct through the existing seam without changing the production default.
    from bilt.core._system import _implementation as implementation_module  # noqa: PLC0415

    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", solver.CausalizationStructuralAnalyzer)
    analyser = implementation.analyser_for(PYTHON_BACKEND)
    request = EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "b"}, Domain.scalar()))
    with pytest.raises(BiltExecutionPlanError, match=r"Cannot compute x.*coupled blocks"):
        analyser.analyse(request)


def _coupled(operations: list[BiltModelConstituent], available: set[str], output: str) -> frozenset[str]:
    structure = validate_request_structure(FullIncidenceGraph(operations), available, output)
    assert structure is not None
    return unavoidable_coupled_outputs(structure.well_determined, output)


def test_boundary_input_breaks_coupling_and_allows_evaluation() -> None:
    assert not _coupled([total, difference], {"a", "y"}, "x")
    analyser = solver.CausalizationStructuralAnalyzer([total, difference], PYTHON_BACKEND)
    with EvaluationContext():
        case = analyser.analyse(EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "y"}, Domain.scalar())))
        expected = 3.0
        assert PYTHON_BACKEND.chain_case(case).callable(a=4.0, y=1.0) == expected


def test_transform_directions_are_respected() -> None:
    @transform
    def y(a: float) -> float:
        return a + 1

    assert not _coupled([total, y], {"a"}, "x")


@pytest.mark.parametrize("nested", (False, True))
def test_directional_feedback_is_coupled(*, nested: bool) -> None:
    @transform
    def x(y: float) -> float:
        return y + 1

    @transform
    def y(x: float) -> float:
        return x + 1

    if nested:
        child = SubSystemConstituent("child", Implementation(Ports, [x, y]))
        assert _coupled([child], set(), "child.x") == frozenset({"child.x", "child.y"})
    else:
        assert _coupled([x, y], set(), "x") == frozenset({"x", "y"})


def test_downstream_coupled_block_does_not_reject_target() -> None:
    @equation
    def source(x: float, a: float) -> bool:
        return x == a

    @equation
    def first(x: float, y: float, z: float) -> bool:
        return y + z == x

    @equation
    def second(y: float, z: float) -> bool:
        return y - z == 1

    assert not _coupled([source, first, second], {"a"}, "x")
    assert _coupled([source, first, second], {"a"}, "y") == frozenset({"y", "z"})


def test_rectangular_scope_with_acyclic_target_is_not_coupled() -> None:
    @equation
    def source(x: float, a: float) -> bool:
        return x == a

    @equation
    def branch(y: float, x: float, z: float) -> bool:
        return y == x + z

    assert not _coupled([source, branch], {"a"}, "x")


def test_incomplete_expansion_is_not_proof_of_coupling() -> None:
    child = SubSystemConstituent("inner", Implementation(Ports, [total, difference]))
    expansion = StructuralExpansion(FullIncidenceGraph([child]), {"inner.a", "inner.b"}, "inner.x")
    snapshot = expansion.snapshot()
    assert not snapshot.complete
    assert not unavoidable_coupled_outputs(snapshot, "inner.x")
    complete = expansion.expand(("inner",))
    assert unavoidable_coupled_outputs(complete, "inner.x") == frozenset({"inner.x", "inner.y"})


def test_directionally_unmatchable_square_scope_falls_back() -> None:
    @transform
    def x(y: float) -> float:
        return y + 1

    snapshot = StructuralExpansion(FullIncidenceGraph([x]), set(), "x").snapshot()
    # Two distinct sources both restricted to x cannot form a perfect assignment.
    snapshot = dataclasses.replace(snapshot, constraints=(*snapshot.constraints, snapshot.constraints[0]))
    assert not unavoidable_coupled_outputs(snapshot, "x")


@pytest.mark.parametrize("size", (2, 3))
def test_small_equation_graphs_agree_with_every_perfect_assignment(size: int) -> None:
    names = tuple(f"v{index}" for index in range(size))
    for bits in itertools.product((False, True), repeat=size * size):
        rows = tuple(frozenset(names[j] for j in range(size) if bits[i * size + j]) for i in range(size))
        snapshot = ExpansionSnapshot(
            constraints=tuple(ExpandedConstraint((), total, row) for row in rows),
            attributes=frozenset(names),
            available=frozenset(),
            unresolved=(),
            expanded_paths=frozenset(),
            query_count=0,
        )
        expected = []
        for assignment in itertools.permutations(names):
            if any(output not in row for output, row in zip(assignment, rows, strict=True)):
                continue
            graph = rx.PyDiGraph(multigraph=False)
            nodes = {name: graph.add_node(name) for name in names}
            for output, row in zip(assignment, rows, strict=True):
                for consumed in row - {output}:
                    graph.add_edge(nodes[consumed], nodes[output], None)
            required = rx.ancestors(graph, nodes[names[0]]) | {nodes[names[0]]}
            expected.append(
                frozenset(
                    graph[node]
                    for component in rx.strongly_connected_components(graph)
                    if len(component) > 1 and required.intersection(component)
                    for node in component
                )
            )
        actual = unavoidable_coupled_outputs(snapshot, names[0])
        if expected:
            assert all(actual == answer for answer in expected)
        else:
            assert not actual
