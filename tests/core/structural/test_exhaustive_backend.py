import typing

import pytest

from bilt import Implementation, Interface, field, system, transform
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent
from bilt.errors import BiltAnalysisError

if typing.TYPE_CHECKING:
    from bilt.core._port_declaration import PortDeclaration


def test_bipartite_graph_shape_for_an_equation() -> None:
    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    parameters = {"v", "i", "r"}
    graph = BipartiteGraph([ohms_law], PYTHON_BACKEND).graph

    for attribute in parameters:
        step = PlanStep(operation=ohms_law, output=attribute)

        assert graph.nodes[attribute]["bipartite"] == "attribute"
        assert graph.nodes[step]["bipartite"] == "constituent"
        assert set(graph.successors(step)) == {attribute}
        assert set(graph.predecessors(step)) == parameters - {attribute}


def test_bipartite_graph_shape_for_a_transform() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    graph = BipartiteGraph([c], PYTHON_BACKEND).graph
    step = PlanStep(operation=c, output="c")

    assert graph.nodes[step]["bipartite"] == "constituent"
    assert set(graph.successors(step)) == {"c"}
    assert set(graph.predecessors(step)) == {"a", "b"}
    assert graph.nodes["a"]["bipartite"] == "attribute"
    assert graph.nodes["b"]["bipartite"] == "attribute"


def test_bipartite_graph_rejects_an_unsupported_constituent() -> None:
    class Unsupported:
        __bilt_name__: str = "unsupported"
        __bilt_parameters__: tuple[str, ...] = ("a",)
        __bilt_port_declarations__: dict[str, PortDeclaration] = {}  # noqa: RUF012

    with pytest.raises(
        NotImplementedError,
        match="is not supported in the bipartite graph",
    ):
        BipartiteGraph([Unsupported()], PYTHON_BACKEND)


def test_search_visits_a_shared_node_from_two_sibling_branches() -> None:
    @transform
    def base(a: float) -> float:
        return a * 2.0

    @transform
    def mid(base: float) -> float:
        return base + 1.0

    @transform
    def top(base: float, mid: float) -> float:
        return base + mid

    plan = BipartiteGraph(
        [base, mid, top],
        PYTHON_BACKEND,
    ).search({"a"}, "top")

    assert plan is not None
    assert [step.output for step in plan] == ["base", "mid", "top"]


def test_search_orders_independent_branches_deterministically() -> None:
    @transform
    def p(a: float) -> float:
        return a + 1.0

    @transform
    def q(b: float) -> float:
        return b + 1.0

    @transform
    def r(p: float, q: float) -> float:
        return p + q

    plan = BipartiteGraph(
        [p, q, r],
        PYTHON_BACKEND,
    ).search({"a", "b"}, "r")

    assert plan is not None
    assert [step.output for step in plan] == ["p", "q", "r"]


def test_search_terminates_on_mutually_derivable_attributes() -> None:
    @equation
    def relation(v: float, i: float) -> bool:
        return v == i * 10.0

    graph = BipartiteGraph([relation], PYTHON_BACKEND)

    assert graph.search({"i"}, "v") == [
        PlanStep(operation=relation, output="v"),
    ]
    assert graph.search({"v"}, "i") == [
        PlanStep(operation=relation, output="i"),
    ]
    assert graph.search(set(), "v") is None


def test_removing_an_unrealizable_step_leaves_its_output_unreachable() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1.0

    graph = BipartiteGraph([square, linear], PYTHON_BACKEND)
    pruned_step = PlanStep(operation=square, output="x")

    graph.remove_failed_plan_step(pruned_step)

    assert pruned_step not in graph.graph
    assert "x" in graph.graph
    assert graph.search({"z"}, "x") is None


def test_search_rejects_an_output_that_is_not_in_the_graph() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    graph = BipartiteGraph([c], PYTHON_BACKEND)

    with pytest.raises(BiltAnalysisError, match="not in the graph"):
        graph.search({"a", "b"}, "missing")


def _resistor_sub_system(name: str = "resistor") -> SubSystemConstituent:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    return SubSystemConstituent(
        name=name,
        implementation=Implementation(Resistor, [ohms_law]),
    )


def _sealed_sub_system(name: str = "sealed") -> SubSystemConstituent:
    @system
    class Sealed(metaclass=Interface):
        a: float = field()
        b: float = field()

    return SubSystemConstituent(
        name=name,
        implementation=Implementation(Sealed, []),
    )


def test_a_sub_system_enters_the_graph_shaped_like_an_equation() -> None:
    sub_system = _resistor_sub_system()
    ports = {"resistor.v", "resistor.i", "resistor.r"}

    graph = BipartiteGraph([sub_system], PYTHON_BACKEND).graph

    for port in ports:
        step = PlanStep(operation=sub_system, output=port)

        assert graph.nodes[port]["bipartite"] == "attribute"
        assert graph.nodes[step]["bipartite"] == "constituent"
        assert set(graph.successors(step)) == {port}
        assert set(graph.predecessors(step)) == ports - {port}


def test_search_solves_a_sub_system_port_in_either_direction() -> None:
    sub_system = _resistor_sub_system()
    graph = BipartiteGraph([sub_system], PYTHON_BACKEND)

    outward = graph.search(
        {"resistor.i", "resistor.r"},
        "resistor.v",
    )
    inward = graph.search(
        {"resistor.v", "resistor.r"},
        "resistor.i",
    )

    assert outward is not None
    assert inward is not None

    assert [(step.operation, step.output, step.child_inputs) for step in outward] == [
        (sub_system, "resistor.v", frozenset({"i", "r"})),
    ]
    assert [(step.operation, step.output, step.child_inputs) for step in inward] == [
        (sub_system, "resistor.i", frozenset({"v", "r"})),
    ]


def test_search_returns_none_when_a_sub_system_port_is_unreachable() -> None:
    graph = BipartiteGraph([_sealed_sub_system()], PYTHON_BACKEND)

    assert graph.search({"sealed.b"}, "sealed.a") is None


def test_search_terminates_on_mutually_derivable_sub_system_ports() -> None:
    graph = BipartiteGraph([_resistor_sub_system()], PYTHON_BACKEND)

    assert graph.search({"resistor.i"}, "resistor.v") is None


def test_removing_a_failed_sub_system_step_keeps_it_while_a_route_exists() -> None:
    sub_system = _resistor_sub_system()
    graph = BipartiteGraph([sub_system], PYTHON_BACKEND)
    step = PlanStep(operation=sub_system, output="resistor.v")

    graph.remove_failed_plan_step(step)

    assert step in graph.graph


def test_removing_a_failed_sub_system_step_removes_it_when_no_route_exists() -> None:
    sub_system = _sealed_sub_system()
    graph = BipartiteGraph([sub_system], PYTHON_BACKEND)
    step = PlanStep(operation=sub_system, output="sealed.a")

    graph.remove_failed_plan_step(step)

    assert step not in graph.graph
