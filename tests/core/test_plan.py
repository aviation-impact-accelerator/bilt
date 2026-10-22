import pytest

from bilt import Implementation, Interface, equation, field, system
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanBlock, PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.symbolic._domains import Domain


@system
class Child(metaclass=Interface):
    a: float = field()
    b: float = field()


@equation
def relation(a: float, b: float) -> bool:
    return b == 2 * a


def test_block_preserves_qualified_child_route_and_operation_identity() -> None:
    child = SubSystemConstituent(name="child", implementation=Implementation(Child, [relation]))
    step = PlanStep(operation=child, output="child.b", child_inputs=frozenset({"a"}))

    block = PlanBlock(steps=(step,))

    assert block.steps[0] is step
    assert block.operations[0] is child
    assert block.outputs == frozenset({"child.b"})
    assert block.inputs == frozenset({"child.a"})
    assert block.steps[0].child_inputs == frozenset({"a"})


def test_plan_preserves_blocks_and_provides_a_flat_inspection_view() -> None:
    first = PlanStep(operation=relation, output="a")
    second = PlanStep(operation=relation, output="b")
    coupled = PlanBlock(steps=(first, second), is_cyclic=True)
    request = EvaluationRequest(output="b", input_domains={})

    plan = AnalysisPlan(request=request, blocks=(coupled,))

    assert plan.blocks[0] is coupled
    assert plan.steps == (first, second)
    assert plan.blocks[0].is_cyclic
    assert plan.output == "b"
    assert plan.inputs == frozenset()


def test_supplied_output_has_an_empty_plan() -> None:
    request = EvaluationRequest(output="a", input_domains={"a": Domain.scalar()})

    plan = AnalysisPlan(request=request, blocks=())

    assert plan.blocks == ()
    assert plan.steps == ()
    assert plan.inputs == frozenset({"a"})
    assert plan.output == "a"


def test_empty_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        PlanBlock(steps=())


def test_multiple_steps_cannot_be_labelled_as_an_acyclic_scc() -> None:
    steps = (PlanStep(operation=relation, output="a"), PlanStep(operation=relation, output="b"))

    with pytest.raises(ValueError, match="exactly one step"):
        PlanBlock(steps=steps)
