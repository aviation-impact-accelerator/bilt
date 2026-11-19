"""Precompilation neutral schedules retain dependency order and source kinds."""

import pytest

from bilt import Interface, field, system
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._program_ir import program_from_plan, program_from_qualified_plan
from bilt.core._program_ir_cache import get_program_cache, program_cache_scope
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock, PlanStep
from bilt.core._structural_analysis._qualified_block import (
    QualifiedBlock,
    QualifiedEquation,
    QualifiedPlan,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.core._transform import transform
from bilt.symbolic._domains import Domain


@equation
def sum_equation(total: float, first: float, second: float) -> bool:
    return total == first + second


@equation
def difference_equation(total: float, first: float, second: float) -> bool:
    return total - first == second


@transform
def double(total: float) -> float:
    return total * 2


@system
class Child(metaclass=Interface):
    value: float = field()
    source: float = field()


def test_plan_ir_is_constructed_without_a_backend() -> None:
    plan = AnalysisPlan(
        request=EvaluationRequest(output="double", input_domains={"first": Domain.scalar(), "second": Domain.scalar()}),
        blocks=(
            PlanBlock((PlanStep(sum_equation, "total"),)),
            PlanBlock((PlanStep(double, "double"),)),
        ),
    )
    program = program_from_plan(plan)
    assert tuple(step.kind for step in program.steps) == ("equation", "transform")
    assert tuple(step.inputs for step in program.steps) == (("first", "second"), ("total",))
    assert tuple(step.output for step in program.steps) == ("total", "double")
    program.verify()


def test_coupled_solve_and_projections_have_explicit_ir_steps() -> None:
    plan = AnalysisPlan(
        request=EvaluationRequest(output="total", input_domains={"second": Domain.scalar()}),
        blocks=(
            PlanBlock(
                (
                    PlanStep(sum_equation, "total"),
                    PlanStep(difference_equation, "first"),
                ),
                is_cyclic=True,
            ),
        ),
    )
    program = program_from_plan(plan)
    assert tuple(step.kind for step in program.steps) == ("linear_solve", "projection", "projection")
    assert program.steps[0].inputs == ("second",)
    assert {step.output for step in program.steps[1:]} == {"first", "total"}
    program.verify()


def test_child_operation_keeps_qualified_port_names() -> None:
    from bilt import Implementation  # noqa: PLC0415

    @equation
    def identity(value: float, source: float) -> bool:
        return value == source

    child = SubSystemConstituent("unit", Implementation(Child, [identity]))
    step = PlanStep(child, "unit.value", child_inputs=frozenset({"source"}))
    plan = AnalysisPlan(
        request=EvaluationRequest(output="unit.value", input_domains={"unit.source": Domain.scalar()}),
        blocks=(PlanBlock((step,)),),
    )
    program = program_from_plan(plan)
    assert program.steps[0].kind == "child"
    assert program.steps[0].inputs == ("unit__source",)
    assert program.steps[0].output == "unit__value"
    program.verify()


def test_qualified_program_preserves_grouped_solve_projection_order() -> None:
    rows = (
        QualifiedEquation(sum_equation),
        QualifiedEquation(difference_equation),
    )
    block = QualifiedBlock(rows=rows, unknowns=("total", "first"))
    plan = QualifiedPlan(
        request=EvaluationRequest(output="first", input_domains={"second": Domain.scalar()}),
        blocks=(block,),
    )
    program = program_from_qualified_plan(plan)
    assert tuple(step.kind for step in program.steps) == ("linear_solve", "projection", "projection")
    assert tuple(step.output for step in program.steps[1:]) == ("total", "first")
    program.verify()


def test_compilation_still_uses_same_python_backend() -> None:
    from bilt._backend._case import BackendCase  # noqa: PLC0415
    from bilt._backend._python import PYTHON_BACKEND  # noqa: PLC0415
    from bilt.core._structural_analysis._realize import realize_plan  # noqa: PLC0415

    plan = AnalysisPlan(
        request=EvaluationRequest(output="double", input_domains={"first": Domain.scalar(), "second": Domain.scalar()}),
        blocks=(
            PlanBlock((PlanStep(sum_equation, "total"),)),
            PlanBlock((PlanStep(double, "double"),)),
        ),
    )
    with EvaluationContext():
        case = realize_plan(plan, PYTHON_BACKEND)
    assert isinstance(case, BackendCase)
    assert PYTHON_BACKEND.chain_case(case).callable(first=2.0, second=3.0) == pytest.approx(10.0)


def test_request_cache_reuses_identical_plan_and_resets_between_requests() -> None:
    plan = AnalysisPlan(
        request=EvaluationRequest(output="total", input_domains={"first": Domain.scalar(), "second": Domain.scalar()}),
        blocks=(PlanBlock((PlanStep(sum_equation, "total"),)),),
    )
    assert get_program_cache() is None
    with program_cache_scope() as cache:
        first = program_from_plan(plan)
        second = program_from_plan(plan)
        assert first is second
        assert len(cache) == 1
        with program_cache_scope() as inner:
            assert inner is cache
            assert program_from_plan(plan) is first
    assert get_program_cache() is None
    with program_cache_scope():
        assert program_from_plan(plan) is not first


def test_request_cache_does_not_alias_different_child_routes() -> None:
    from bilt import Implementation  # noqa: PLC0415

    @equation
    def identity(value: float, source: float) -> bool:
        return value == source

    child = SubSystemConstituent("unit", Implementation(Child, [identity]))
    request = EvaluationRequest(
        output="unit.value",
        input_domains={"unit.source": Domain.scalar(), "unit.value": Domain.scalar()},
    )
    first = AnalysisPlan(request, (PlanBlock((PlanStep(child, "unit.value", frozenset({"source"})),)),))
    other = AnalysisPlan(request, (PlanBlock((PlanStep(child, "unit.value", frozenset({"value"})),)),))
    with program_cache_scope() as cache:
        first_ir = program_from_plan(first)
        other_ir = program_from_plan(other)
        assert first_ir is not other_ir
        assert first_ir.steps[0].inputs != other_ir.steps[0].inputs
        assert len(cache) == len((first, other))


def test_request_cache_does_not_alias_different_equation_choices() -> None:
    request = EvaluationRequest(output="total", input_domains={"first": Domain.scalar(), "second": Domain.scalar()})
    first = AnalysisPlan(request, (PlanBlock((PlanStep(sum_equation, "total"),)),))
    other = AnalysisPlan(request, (PlanBlock((PlanStep(difference_equation, "total"),)),))
    with program_cache_scope() as cache:
        a = program_from_plan(first)
        b = program_from_plan(other)
        assert a is not b
        assert len(cache) == len((first, other))


def test_qualified_request_cache_reuses_identical_blocks() -> None:
    block = QualifiedBlock(
        (QualifiedEquation(sum_equation), QualifiedEquation(difference_equation)),
        ("total", "first"),
    )
    request = EvaluationRequest(output="first", input_domains={"second": Domain.scalar()})
    a = QualifiedPlan(request, (block,))
    b = QualifiedPlan(request, (block,))
    with program_cache_scope():
        assert program_from_qualified_plan(a) is program_from_qualified_plan(b)
