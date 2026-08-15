from bilt.core._equation import equation
from bilt.core._structural_analysis._plan import CoupledBlock
from bilt.core._structural_analysis._scc._bipartite import BipartiteGraph


def test_scc_analysis_groups_coupled_equations() -> None:
    @equation
    def first(x: float, y: float, a: float) -> bool:
        return x + y == a

    @equation
    def second(x: float, y: float, b: float) -> bool:
        return x - y == b

    graph = BipartiteGraph((first, second))

    plan = graph.find_plan(
        available={"a", "b"},
        output="x",
    )

    assert plan is not None
    assert plan.inputs == frozenset({"a", "b"})
    assert plan.output == "x"
    assert len(plan.steps) == 1

    block = plan.steps[0]

    assert isinstance(block, CoupledBlock)
    assert block.outputs == frozenset({"x", "y"})
    assert {step.operation for step in block.steps} == {first, second}
