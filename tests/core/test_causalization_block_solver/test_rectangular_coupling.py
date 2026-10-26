import dataclasses
import typing

import pytest

from bilt import Implementation, Interface, equation, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._causalization_block_solver import _backend as solver
from bilt.core._structural_analysis._causalization_block_solver import _planning as planning
from bilt.core._structural_analysis._causalization_block_solver import _search as search
from bilt.core._structural_analysis._causalization_block_solver._coupling import unavoidable_coupled_outputs
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltExecutionPlanError,
    BiltOverconstrainedError,
    BiltUnderconstrainedError,
)
from bilt.core._structural_analysis._causalization_block_solver._expand_structure import StructuralExpansion
from bilt.core._structural_analysis._causalization_block_solver._full_incidence_graph import FullIncidenceGraph
from bilt.core._structural_analysis._causalization_block_solver._structural_validation import validate_request_structure
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._system import _implementation as implementation_module
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import collections.abc

    import rustworkx as rx

    from bilt.core._structural_analysis._causalization_block_solver._dm import DMPartition


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


@equation
def surplus(u: float, v: float, x: float) -> bool:
    return u + v == x


@equation
def explicit(x: float, a: float, b: float) -> bool:
    return x == a + b


@equation
def input_a(a: float, inner: Ports) -> bool:
    return inner.a == a


@equation
def input_b(b: float, inner: Ports) -> bool:
    return inner.b == b


@equation
def output_x(x: float, inner: Ports) -> bool:
    return x == inner.x


@equation
def left_a(a: float, left: Ports) -> bool:
    return left.a == a


@equation
def left_b(b: float, left: Ports) -> bool:
    return left.b == b


@equation
def right_a(a: float, left: Ports, right: Ports) -> bool:
    return right.a == left.x + a


@equation
def right_b(b: float, right: Ports) -> bool:
    return right.b == b


@equation
def combined(x: float, left: Ports, right: Ports) -> bool:
    return x == left.x + right.x


def _nested(depth: int, *, siblings: bool) -> Implementation:
    implementation = Implementation(Ports, [total, difference, surplus])
    for _ in range(depth):
        if siblings:
            children = [SubSystemConstituent(name, implementation) for name in ("left", "right")]
            implementation = Implementation(Ports, [*children, left_a, left_b, right_a, right_b, combined])
        else:
            inner = SubSystemConstituent("inner", implementation)
            implementation = Implementation(Ports, [inner, input_a, input_b, output_x])
    return implementation


def _unexpected(*_: object, **__: object) -> typing.Never:
    msg = "Rectangular coupled requests must be rejected before execution matching, settlement or realisation."
    raise AssertionError(msg)


@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("siblings", (False, True))
def test_rectangular_nested_coupling_skips_execution_search(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    *,
    siblings: bool,
) -> None:
    implementation = _nested(depth, siblings=siblings)
    monkeypatch.setattr(implementation_module, "ExhaustiveStructuralAnalyser", solver.CausalizationStructuralAnalyzer)
    for owner, name in ((search, "find_matching"), (planning, "settle_assignments"), (solver, "realize_plan")):
        monkeypatch.setattr(owner, name, _unexpected)
    monkeypatch.setattr(SubSystemConstituent, "required_inputs_for", _unexpected)
    request = EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "b"}, Domain.scalar()))
    with EvaluationContext(), pytest.raises(BiltExecutionPlanError, match=r"Cannot compute x.*coupled blocks"):
        implementation.analyser_for(PYTHON_BACKEND).analyse(request)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("same_labels", (False, True))
def test_validation_keeps_full_scope_and_reuses_its_dm_partition(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reverse: bool,
    same_labels: bool,
) -> None:
    from bilt.core._structural_analysis._causalization_block_solver import (  # noqa: PLC0415
        _structural_validation as validation,
    )

    original = validation.dulmage_mendelsohn
    calls = 0

    def counted(graph: rx.PyGraph, attributes: collections.abc.Set[int]) -> DMPartition:
        nonlocal calls
        calls += 1
        return original(graph, attributes)

    monkeypatch.setattr(validation, "dulmage_mendelsohn", counted)
    operations = [total, difference, surplus]
    if same_labels:
        for operation in operations:
            monkeypatch.setattr(operation, "__bilt_name__", "shared")
    if reverse:
        operations.reverse()
    validated = validate_request_structure(FullIncidenceGraph(operations), {"a", "b"}, "x")
    assert validated is not None
    full, core = validated.snapshot, validated.well_determined
    assert full.complete
    assert full.attributes - full.available == {"x", "y", "u", "v"}
    assert core.attributes - core.available == {"x", "y"}
    assert {id(row.operation) for row in full.constraints} == {id(total), id(difference), id(surplus)}
    assert {id(row.operation) for row in core.constraints} == {id(total), id(difference)}
    assert unavoidable_coupled_outputs(core, "x") == frozenset({"x", "y"})
    assert calls == 1


def test_rectangular_acyclic_route_still_executes() -> None:
    analyser = solver.CausalizationStructuralAnalyzer([explicit, surplus], PYTHON_BACKEND)
    request = EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "b"}, Domain.scalar()))
    with EvaluationContext():
        case = analyser.analyse(request)
        expected = 6.0
        assert PYTHON_BACKEND.chain_case(case).callable(a=4.0, b=2.0) == expected


def test_rectangular_downstream_cycle_does_not_reject_target() -> None:
    @equation
    def first(x: float, y: float, z: float) -> bool:
        return y + z == x

    @equation
    def second(y: float, z: float) -> bool:
        return y - z == 1

    operations = [explicit, first, second, surplus]
    validated = validate_request_structure(FullIncidenceGraph(operations), {"a", "b"}, "x")
    assert validated is not None
    assert not unavoidable_coupled_outputs(validated.well_determined, "x")
    assert unavoidable_coupled_outputs(validated.well_determined, "y") == frozenset({"y", "z"})
    analyser = solver.CausalizationStructuralAnalyzer(operations, PYTHON_BACKEND)
    with EvaluationContext():
        case = analyser.analyse(EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "b"}, Domain.scalar())))
        expected = 6.0
        assert PYTHON_BACKEND.chain_case(case).callable(a=4.0, b=2.0) == expected


def test_competing_constraints_are_not_hidden_by_core_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def competing(x: float, a: float) -> bool:
        return x == a

    monkeypatch.setattr(solver, "unavoidable_coupled_outputs", _unexpected)
    analyser = solver.CausalizationStructuralAnalyzer([explicit, competing, surplus], PYTHON_BACKEND)
    request = EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "b"}, Domain.scalar()))
    with pytest.raises(BiltOverconstrainedError):
        analyser.analyse(request)


def test_underdetermined_target_is_not_a_coupled_execution_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(solver, "unavoidable_coupled_outputs", _unexpected)
    analyser = solver.CausalizationStructuralAnalyzer([surplus], PYTHON_BACKEND)
    with pytest.raises(BiltUnderconstrainedError):
        analyser.analyse(EvaluationRequest(output="x", input_domains={}))


def test_transform_direction_can_prevent_a_core_proof() -> None:
    @transform
    def x(y: float) -> float:
        return y + 1

    validated = validate_request_structure(FullIncidenceGraph([x, explicit, surplus]), {"a", "b"}, "x")
    assert validated is not None
    assert not unavoidable_coupled_outputs(validated.well_determined, "x")

    analyser = solver.CausalizationStructuralAnalyzer([x, explicit, surplus], PYTHON_BACKEND)
    with EvaluationContext():
        case = analyser.analyse(EvaluationRequest(output="x", input_domains=dict.fromkeys({"a", "b"}, Domain.scalar())))
        expected = 6.0
        assert PYTHON_BACKEND.chain_case(case).callable(a=4.0, b=2.0) == expected


def test_directional_feedback_is_found_in_a_rectangular_core() -> None:
    @transform
    def x(y: float) -> float:
        return y + 1

    @transform
    def y(x: float) -> float:
        return x + 1

    validated = validate_request_structure(FullIncidenceGraph([x, y, surplus]), set(), "x")
    assert validated is not None
    assert unavoidable_coupled_outputs(validated.well_determined, "x") == frozenset({"x", "y"})


def test_incomplete_rectangular_child_cannot_prove_coupling() -> None:
    child = SubSystemConstituent("inner", Implementation(Ports, [total, difference, surplus]))
    expansion = StructuralExpansion(FullIncidenceGraph([child]), {"inner.a", "inner.b"}, "inner.x")
    snapshot = expansion.snapshot()
    assert not snapshot.complete
    assert not unavoidable_coupled_outputs(snapshot, "inner.x")


def test_missing_core_dependency_cannot_prove_coupling() -> None:
    validated = validate_request_structure(FullIncidenceGraph([total, difference, surplus]), {"a", "b"}, "x")
    assert validated is not None
    core = validated.well_determined
    incomplete = dataclasses.replace(
        core,
        constraints=(
            dataclasses.replace(core.constraints[0], attributes=core.constraints[0].attributes | {"missing"}),
            *core.constraints[1:],
        ),
    )
    assert not unavoidable_coupled_outputs(incomplete, "x")


def test_shared_children_preserve_distinct_primitive_rows() -> None:
    implementation = _nested(1, siblings=True)
    validated = validate_request_structure(implementation.structural_analysis()._incidence, {"a", "b"}, "x")  # noqa: SLF001
    assert validated is not None
    rows = [row for row in validated.well_determined.constraints if row.operation is total]
    assert {row.path for row in rows} == {("left",), ("right",)}
    assert len(rows) == len({row.path for row in rows})
