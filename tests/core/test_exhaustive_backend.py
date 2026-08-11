import typing

import pytest

import bilt.core._realize_analysis._realize as realize_module
from bilt._backend._python._backend import PythonBackend
from bilt.core import Implementation, Interface, field, system, transform
from bilt.core._equation import equation
from bilt.core._realize_analysis._realize import RealizationFailure, _operation_to_callable
from bilt.core._structural_analysis import BiltSystemAnalysisError, StructuralAnalyser
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph, required_inputs_for_target
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._sub_system import SubSystemConstituent


@typing.runtime_checkable
class _RuntimeStructuralAnalyser(StructuralAnalyser, typing.Protocol):
    pass


def test_exhaustive_analyser_conforms_to_structural_analyser_protocol() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c])
    assert isinstance(analyser, _RuntimeStructuralAnalyser)


def test_exhaustive_backend_analyser_reuses_graph() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([c])

    graph_before = analyser.bipartite_graph
    analyser.analyse({"a", "b"}, "c", backend)
    analyser.analyse({"a", "b"}, "c", backend)

    assert analyser.bipartite_graph is graph_before


def test_exhaustive_backend_can_be_called() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([c])
    result = analyser.analyse({"a", "b"}, "c", backend)
    backend_case = backend.chain_case(result)
    assert backend_case.callable(a=1.0, b=2.0) == c(a=1.0, b=2.0)


def test_exhaustive_analyser_required_inputs_with_path() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c])
    required = analyser.required_inputs_for_target({"a", "b"}, "c")
    assert required == frozenset({"a", "b"})


def test_exhaustive_analyser_required_inputs_no_path() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c])
    required = analyser.required_inputs_for_target({"a"}, "c")
    assert required is None


def test_bipartite_graph_contains_all_sub_system_ports() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])
    sub_system = SubSystemConstituent(__bilt_name__="resistor", interface=Resistor, implementation=resistor_impl)

    graph = BipartiteGraph([sub_system])

    # Verify all ports are present as attribute nodes
    assert "resistor.v" in graph.graph
    assert "resistor.i" in graph.graph
    assert "resistor.r" in graph.graph
    assert graph.graph.nodes["resistor.v"]["bipartite"] == "attribute"
    assert graph.graph.nodes["resistor.i"]["bipartite"] == "attribute"
    assert graph.graph.nodes["resistor.r"]["bipartite"] == "attribute"


def test_bipartite_graph_search_returns_plan_with_sub_system_step() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])
    sub_system = SubSystemConstituent(__bilt_name__="resistor", interface=Resistor, implementation=resistor_impl)

    graph = BipartiteGraph([sub_system])

    # Search for resistor.v with i and r available
    plan = graph.search({"resistor.i", "resistor.r"}, "resistor.v")

    assert plan is not None
    last_step = plan[-1]
    assert isinstance(last_step, PlanStep)
    assert last_step.operation is sub_system
    assert last_step.output == "resistor.v"


def test_bipartite_graph_sub_system_plan_includes_required_inputs() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])
    sub_system = SubSystemConstituent(__bilt_name__="resistor", interface=Resistor, implementation=resistor_impl)

    graph = BipartiteGraph([sub_system])

    # Search for resistor.v with resistor.i and resistor.r available
    # The plan should include a sub-system step and verify it requires the available inputs
    plan = graph.search({"resistor.i", "resistor.r"}, "resistor.v")

    assert plan is not None
    outputs = {step.output for step in plan}
    assert "resistor.v" in outputs

    # Verify the sub-system step requires exactly the available ports
    sub_system_step = next(s for s in plan if s.output == "resistor.v")
    operation = sub_system_step.operation
    assert isinstance(operation, SubSystemConstituent)
    assert required_inputs_for_target(PlanStep(operation=operation, output="resistor.v")) == {
        "resistor.i",
        "resistor.r",
    }


def test_bipartite_graph_sub_system_no_path_returns_none() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])
    sub_system = SubSystemConstituent(__bilt_name__="resistor", interface=Resistor, implementation=resistor_impl)

    graph = BipartiteGraph([sub_system])

    # Search for resistor.v with only resistor.i available (r is missing)
    plan = graph.search({"resistor.i"}, "resistor.v")

    # Should return None because r is not available and no transform can produce it
    assert plan is None


def test_bipartite_graph_sub_system_mutual_derivability_terminates() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()

    @equation
    def relation(v: float, i: float) -> bool:
        return v == i * 10

    resistor_impl = Implementation(Resistor, [relation])
    sub_system = SubSystemConstituent(__bilt_name__="resistor", interface=Resistor, implementation=resistor_impl)

    graph = BipartiteGraph([sub_system])

    # Search for resistor.v with only resistor.i available
    plan = graph.search({"resistor.i"}, "resistor.v")

    # Should return a valid plan without infinite recursion
    assert plan is not None
    outputs = {step.output for step in plan}
    assert "resistor.v" in outputs

    # Search for resistor.i with only resistor.v available
    plan2 = graph.search({"resistor.v"}, "resistor.i")

    # Should return a valid plan without infinite recursion
    assert plan2 is not None
    outputs2 = {step.output for step in plan2}
    assert "resistor.i" in outputs2


def test_realization_of_sub_system_step_mangles_dotted_names() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: float, resistor: Resistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: float, resistor: Resistor) -> bool:
        return resistance == resistor.r

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor_impl},
    )

    backend = PythonBackend()
    planned_case = component_impl.analyser.analyse({"current", "resistance"}, "voltage", backend)

    # Verify that no parameters contain dots in the plan (they should be mangled)
    for intermediate in planned_case.intermediates:
        for param in intermediate.parameters:
            assert "." not in param, f"Parameter {param!r} contains a dot and was not mangled"


def test_realization_of_sub_system_step_produces_correct_result() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])

    @system
    class Component(metaclass=Interface):
        voltage: float = field()
        current: float = field()
        resistance: float = field()

    @equation
    def voltage_eq(voltage: float, resistor: Resistor) -> bool:
        return voltage == resistor.v

    @equation
    def current_eq(current: float, resistor: Resistor) -> bool:
        return current == resistor.i

    @equation
    def resistance_eq(resistance: float, resistor: Resistor) -> bool:
        return resistance == resistor.r

    component_impl = Implementation(
        Component,
        [voltage_eq, current_eq, resistance_eq],
        internal_implementations={"resistor": resistor_impl},
    )

    backend = PythonBackend()
    final_callable = component_impl.realized_callable(frozenset({"current", "resistance"}), "voltage", backend)

    # Invoke the chained callable with test values
    expected_result = 6.0
    result = final_callable.callable(current=2.0, resistance=3.0)
    assert result == expected_result


def test_realization_raises_on_bogus_sub_system_port() -> None:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    resistor_impl = Implementation(Resistor, [ohms_law])
    sub_system = SubSystemConstituent(__bilt_name__="resistor", interface=Resistor, implementation=resistor_impl)

    # A PlanStep naming a port this sub-system doesn't have is a construction bug, not a
    # backend limitation -- it must raise rather than degrade to a routable RealizationFailure.
    bogus_step = PlanStep(operation=sub_system, output="resistor.unknown_port")

    with pytest.raises(BiltSystemAnalysisError):
        _operation_to_callable(bogus_step, PythonBackend())


def test_realization_failure_on_unsolvable_sub_system_step() -> None:
    @system
    class EmptySystem(metaclass=Interface):
        a: float = field()
        b: float = field()

    empty_impl = Implementation(EmptySystem, [])
    sub_system = SubSystemConstituent(__bilt_name__="subsys", interface=EmptySystem, implementation=empty_impl)

    # subsys.a is a real port, but the inner implementation has no equations, so there is
    # genuinely no path to it -- this must degrade gracefully to a routable RealizationFailure.
    unsolvable_step = PlanStep(operation=sub_system, output="subsys.a")

    result = _operation_to_callable(unsolvable_step, PythonBackend())

    assert isinstance(result, RealizationFailure)


def test_no_common_names_heuristic_remains() -> None:
    assert not hasattr(realize_module, "_COMMON_NAMES")
    assert not hasattr(realize_module, "_map_parent_inputs_to_subsystem")


def test_realization_failure_does_not_mutate_shared_graph() -> None:
    """A step that cannot be realized for one query must stay available to later ones.

    Realizability depends on the requested inputs and the backend, and the analyser's graph is
    cached for the life of the implementation, so exclusions have to be scoped to one query.
    """

    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def offset(z: float, y: float) -> bool:
        return z == y + 1.0

    backend = PythonBackend()
    analyser = ExhaustiveStructuralAnalyser([square, offset])
    nodes_before = analyser.bipartite_graph.graph.number_of_nodes()

    # `z == x * x` cannot be isolated for x, so realization of that step fails.
    with pytest.raises(BiltSystemAnalysisError):
        analyser.analyse({"z"}, "x", backend)

    assert analyser.bipartite_graph.graph.number_of_nodes() == nodes_before

    # The unrelated route to z must still be reachable afterwards.
    expected_z = 5.0
    case = analyser.analyse({"y"}, "z", backend)
    assert backend.chain_case(case).callable(y=4.0) == expected_z
