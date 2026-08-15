from bilt.core._equation import equation
from bilt.core._structural_analysis._plan import CoupledBlock
from bilt.core._structural_analysis._scc._bipartite import BipartiteGraph
from bilt.core._structural_analysis._scc._coupled import analyse_coupled_block


def test_analyse_coupled_block_as_linear_system() -> None:
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

    expected_length = 1

    assert plan is not None
    assert len(plan.steps) == expected_length

    block = plan.steps[0]
    assert isinstance(block, CoupledBlock)

    system = analyse_coupled_block(block)

    expected = 2

    assert set(system.unknowns) == {"x", "y"}
    assert len(system.coefficients) == expected
    assert len(system.rhs) == expected

    assert {tree.head.value for tree in system.rhs} == {"a", "b"}
