import typing

import pytest

from bilt._backend._errors import UnsupportedNodeError
from bilt._backend._python._backend import PythonBackend
from bilt.core import BiltSystemAnalysisError, transform
from bilt.core._equation import equation
from bilt.core._realize_analysis._realize import RealizationFailure
from bilt.core._structural_analysis import StructuralAnalyser
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph, required_inputs_for_target
from bilt.core._structural_analysis._plan import PlanStep
from bilt.symbolic._isolation._errors import IsolationError


def _refuse_to_compile(_function_definition: object) -> typing.NoReturn:
    """Stand in for a backend meeting a construct it does not support."""
    msg = "AST node 'While' is unsupported by this backend."
    raise UnsupportedNodeError(msg)


def test_exhaustive_analyser_conforms_to_the_structural_analyser_protocol() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c])

    assert isinstance(analyser, StructuralAnalyser)


def test_exhaustive_analyser_produces_a_working_callable() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    backend = PythonBackend()
    result = ExhaustiveStructuralAnalyser([c]).analyse({"a", "b"}, "c", backend)
    backend_case = backend.chain_case(result)
    assert backend_case.callable(a=1.0, b=2.0) == c(a=1.0, b=2.0)


def test_required_inputs_for_target_of_a_transform() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    assert required_inputs_for_target(PlanStep(operation=c, output="c")) == frozenset({"a", "b"})


def test_required_inputs_for_target_of_an_equation() -> None:
    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    assert required_inputs_for_target(PlanStep(operation=ohms_law, output="v")) == frozenset({"i", "r"})


def test_required_inputs_for_target_rejects_a_transform_output_mismatch() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    with pytest.raises(BiltSystemAnalysisError, match="does not produce"):
        required_inputs_for_target(PlanStep(operation=c, output="a"))


def test_required_inputs_for_target_rejects_an_unknown_equation_parameter() -> None:
    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    with pytest.raises(BiltSystemAnalysisError, match="does not have parameter"):
        required_inputs_for_target(PlanStep(operation=ohms_law, output="p"))


def test_required_inputs_for_target_rejects_an_unsupported_operation() -> None:
    class Unsupported:
        __bilt_name__: str = "unsupported"
        __bilt_parameters__: tuple[str, ...] = ("a",)

    with pytest.raises(NotImplementedError, match="is not supported"):
        required_inputs_for_target(PlanStep(operation=Unsupported(), output="unsupported"))


def test_bipartite_graph_shape_for_an_equation() -> None:
    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    parameters = {"v", "i", "r"}
    graph = BipartiteGraph([ohms_law]).graph

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

    graph = BipartiteGraph([c]).graph
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

    with pytest.raises(NotImplementedError, match="is not supported in the bipartite graph"):
        BipartiteGraph([Unsupported()])


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

    plan = BipartiteGraph([base, mid, top]).search({"a"}, "top")

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

    plan = BipartiteGraph([p, q, r]).search({"a", "b"}, "r")

    assert plan is not None
    assert [step.output for step in plan] == ["p", "q", "r"]


def test_search_terminates_on_mutually_derivable_attributes() -> None:
    @equation
    def relation(v: float, i: float) -> bool:
        return v == i * 10.0

    graph = BipartiteGraph([relation])
    no_inputs: set[str] = set()

    assert graph.search({"i"}, "v") == [PlanStep(operation=relation, output="v")]
    assert graph.search({"v"}, "i") == [PlanStep(operation=relation, output="i")]
    assert graph.search(no_inputs, "v") is None


def test_search_routes_around_an_excluded_plan_step_without_mutating_the_graph() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1.0

    graph = BipartiteGraph([square, linear])
    nodes_before = set(graph.graph.nodes)
    excluded_step = PlanStep(operation=square, output="x")

    assert graph.search({"z", "w"}, "x", excluded=frozenset({excluded_step})) == [
        PlanStep(operation=linear, output="x")
    ]
    assert set(graph.graph.nodes) == nodes_before
    assert graph.search({"z", "w"}, "x") == [excluded_step]


def test_removing_an_unrealizable_plan_step_leaves_its_output_present_but_unreachable() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1.0

    graph = BipartiteGraph([square, linear])
    pruned_step = PlanStep(operation=square, output="x")
    graph.remove_failed_plan_step(pruned_step)

    assert pruned_step not in graph.graph
    assert "x" in graph.graph
    assert graph.search({"z"}, "x") is None


def test_search_rejects_an_output_that_is_not_in_the_graph() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    graph = BipartiteGraph([c])

    with pytest.raises(BiltSystemAnalysisError, match="not in the graph"):
        graph.search({"a", "b"}, "missing")


def test_analyser_prunes_a_permanently_unrealizable_step_and_keeps_it_pruned() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1.0

    expected_x = 5.0
    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([square, linear])
    pruned_step = PlanStep(operation=square, output="x")

    first_case = backend.chain_case(analyser.analyse({"z", "w"}, "x", backend))
    assert first_case.callable(w=4.0, z=25.0) == expected_x
    assert pruned_step not in analyser.bipartite_graph.graph

    second_case = backend.chain_case(analyser.analyse({"z", "w"}, "x", backend))
    assert second_case.callable(w=4.0, z=25.0) == expected_x
    assert pruned_step not in analyser.bipartite_graph.graph


def test_analyser_raises_when_no_route_survives_pruning_but_other_routes_still_work() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def offset(z: float, y: float) -> bool:
        return z == y + 1.0

    expected_z = 5.0
    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([square, offset])

    with pytest.raises(BiltSystemAnalysisError, match="Cannot compute x"):
        analyser.analyse({"z"}, "x", backend)

    surviving_case = backend.chain_case(analyser.analyse({"y"}, "z", backend))
    assert surviving_case.callable(y=4.0) == expected_z


def test_a_backend_rejection_is_not_pruned_from_the_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def from_w(x: float, w: float) -> bool:
        return x == w + 1.0

    expected_x = 5.0
    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([from_w])
    nodes_before = set(analyser.bipartite_graph.graph.nodes)

    monkeypatch.setattr(backend, "transpile", _refuse_to_compile)
    with pytest.raises(BiltSystemAnalysisError):
        analyser.analyse({"w"}, "x", backend)

    assert set(analyser.bipartite_graph.graph.nodes) == nodes_before

    monkeypatch.undo()
    surviving_case = backend.chain_case(analyser.analyse({"w"}, "x", backend))
    assert surviving_case.callable(w=4.0) == expected_x


def test_exhausted_search_reports_why_the_last_route_was_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    @equation
    def from_w(x: float, w: float) -> bool:
        return x == w + 1.0

    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([from_w])
    monkeypatch.setattr(backend, "transpile", _refuse_to_compile)

    with pytest.raises(BiltSystemAnalysisError, match="unsupported") as raised:
        analyser.analyse({"w"}, "x", backend)

    assert isinstance(raised.value.__cause__, UnsupportedNodeError)


def test_exhausted_search_reports_an_isolation_failure_as_the_cause() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    analyser = ExhaustiveStructuralAnalyser([square])

    with pytest.raises(BiltSystemAnalysisError) as raised:
        analyser.analyse({"z"}, "x", PythonBackend())

    assert isinstance(raised.value.__cause__, IsolationError)


def test_realization_failure_is_permanent_by_default() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    assert RealizationFailure(failed_plan_step=PlanStep(operation=square, output="x")).permanent is True
