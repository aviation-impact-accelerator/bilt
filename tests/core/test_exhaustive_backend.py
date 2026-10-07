import typing

import pytest

from bilt import Implementation, Interface, field, system, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis import StructuralAnalyser
from bilt.core._structural_analysis._exhaustive import ExhaustiveStructuralAnalyser
from bilt.core._structural_analysis._exhaustive._bipartite import BipartiteGraph
from bilt.core._structural_analysis._plan import PlanStep
from bilt.core._structural_analysis._realize import RealizationFailure, _operation_to_callable
from bilt.core._sub_system import SubSystemConstituent
from bilt.errors import BiltAnalysisError, BiltIsolationError, BiltUnsupportedNodeError
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend._base_backend import Backend
    from bilt._backend._callable import BackendCallable
    from bilt._backend._case import BackendCase
    from bilt.units import Dimension


@typing.runtime_checkable
class _RuntimeStructuralAnalyser(StructuralAnalyser, typing.Protocol):
    """An isinstance-able view of the protocol, so it need not be runtime checkable in the source."""


SCALAR_DOMAIN = Domain.scalar()


def _analyse_scalar(
    analyser: StructuralAnalyser,
    available: set[str],
    output: str,
) -> BackendCase:
    request = EvaluationRequest(
        output=output,
        input_domains=dict.fromkeys(available, SCALAR_DOMAIN),
    )

    with EvaluationContext():
        return analyser.analyse(request)


def _refuse_to_compile(_function_definition: object) -> typing.NoReturn:
    """Stand in for a backend meeting a construct it does not support."""
    msg = "AST node 'While' is unsupported by this backend."
    raise BiltUnsupportedNodeError(msg)


def _refuse_one_transform(
    monkeypatch: pytest.MonkeyPatch,
    backend: Backend,
    refused: str,
) -> list[str]:
    """Make ``backend`` reject one named function, and return the log of what it was asked to compile.

    The log is what makes a re-attempt visible: a backend that is asked twice for a function it
    has already refused has been told nothing by the first refusal.
    """
    attempts: list[str] = []
    transpile = backend.transpile

    def counting_transpile(function_definition: ast.FunctionDef) -> BackendCallable:
        attempts.append(function_definition.name)
        if function_definition.name == refused:
            _refuse_to_compile(function_definition)
        return transpile(function_definition)

    monkeypatch.setattr(backend, "transpile", counting_transpile)
    return attempts


def test_exhaustive_analyser_conforms_to_the_structural_analyser_protocol() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c], PYTHON_BACKEND)

    assert isinstance(analyser, _RuntimeStructuralAnalyser)


def test_exhaustive_analyser_produces_a_working_callable() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    result = _analyse_scalar(ExhaustiveStructuralAnalyser([c], PYTHON_BACKEND), {"a", "b"}, "c")
    backend_case = PYTHON_BACKEND.chain_case(result)

    assert backend_case.callable(a=1.0, b=2.0) == c(a=1.0, b=2.0)
    assert result.intermediates[-1].output_domain == SCALAR_DOMAIN


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
        __bilt_port_declarations__: dict[str, Dimension] = {}  # noqa: RUF012

    with pytest.raises(NotImplementedError, match="is not supported in the bipartite graph"):
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

    plan = BipartiteGraph([base, mid, top], PYTHON_BACKEND).search({"a"}, "top")

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

    plan = BipartiteGraph([p, q, r], PYTHON_BACKEND).search({"a", "b"}, "r")

    assert plan is not None
    assert [step.output for step in plan] == ["p", "q", "r"]


def test_search_terminates_on_mutually_derivable_attributes() -> None:
    @equation
    def relation(v: float, i: float) -> bool:
        return v == i * 10.0

    graph = BipartiteGraph([relation], PYTHON_BACKEND)
    no_inputs: set[str] = set()

    assert graph.search({"i"}, "v") == [PlanStep(operation=relation, output="v")]
    assert graph.search({"v"}, "i") == [PlanStep(operation=relation, output="i")]
    assert graph.search(no_inputs, "v") is None


def test_removing_an_unrealizable_plan_step_leaves_its_output_present_but_unreachable() -> None:
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


def test_analyser_prunes_an_unrealizable_step_and_keeps_it_pruned() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1.0

    expected_x = 5.0
    analyser = ExhaustiveStructuralAnalyser([square, linear], PYTHON_BACKEND)
    pruned_step = PlanStep(operation=square, output="x")

    first_case = PYTHON_BACKEND.chain_case(_analyse_scalar(analyser, {"z", "w"}, "x"))
    assert first_case.callable(w=4.0, z=25.0) == expected_x
    assert pruned_step not in analyser.bipartite_graph.graph

    second_case = PYTHON_BACKEND.chain_case(_analyse_scalar(analyser, {"z", "w"}, "x"))
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
    analyser = ExhaustiveStructuralAnalyser([square, offset], PYTHON_BACKEND)

    with pytest.raises(BiltAnalysisError, match="Cannot compute x"):
        _analyse_scalar(analyser, {"z"}, "x")

    surviving_case = PYTHON_BACKEND.chain_case(_analyse_scalar(analyser, {"y"}, "z"))
    assert surviving_case.callable(y=4.0) == expected_z


def test_a_step_one_backend_refuses_is_only_attempted_once_across_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @transform
    def b(a: float) -> float:
        return a * 2.0

    @transform
    def c(b: float) -> float:
        return b + 1.0

    @transform
    def d(b: float) -> float:
        return b + 2.0

    analyser = ExhaustiveStructuralAnalyser([b, c, d], PYTHON_BACKEND)
    attempts = _refuse_one_transform(monkeypatch, PYTHON_BACKEND, "b")

    for output in ("c", "d"):
        with pytest.raises(BiltAnalysisError):
            _analyse_scalar(analyser, {"a"}, output)

    assert attempts == ["b"]


def test_a_step_one_backend_refuses_stays_available_to_another_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @system
    class Doubler(metaclass=Interface):
        a: float = field()
        c: float = field()

    @transform
    def c(a: float) -> float:
        return a * 2.0

    expected_c = 4.0
    implementation = Implementation(Doubler, [c])
    request = EvaluationRequest(
        output="c",
        input_domains={"a": SCALAR_DOMAIN},
    )

    _refuse_one_transform(monkeypatch, PYTHON_BACKEND, "c")
    with EvaluationContext():
        with pytest.raises(BiltAnalysisError):
            implementation.get_evaluation(request, PYTHON_BACKEND)

        surviving = implementation.get_evaluation(request, JAX_BACKEND)

    assert surviving.callable(a=2.0) == expected_c


def test_exhausted_search_reports_why_the_last_route_was_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @equation
    def from_w(x: float, w: float) -> bool:
        return x == w + 1.0

    analyser = ExhaustiveStructuralAnalyser([from_w], PYTHON_BACKEND)
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", _refuse_to_compile)

    with pytest.raises(BiltAnalysisError, match="unsupported") as raised:
        _analyse_scalar(analyser, {"w"}, "x")

    assert isinstance(raised.value.__cause__, BiltUnsupportedNodeError)


def test_exhausted_search_reports_an_isolation_failure_as_the_cause() -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    analyser = ExhaustiveStructuralAnalyser([square], PYTHON_BACKEND)

    with pytest.raises(BiltAnalysisError) as raised:
        _analyse_scalar(analyser, {"z"}, "x")

    assert isinstance(raised.value.__cause__, BiltIsolationError)


def _resistor_sub_system(name: str = "resistor") -> SubSystemConstituent:
    """A sub-system wrapping a resistor whose one equation solves for any of its three ports."""

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
    """A sub-system with real ports but no equations, so no port can be reached from any other."""

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

    outward = graph.search({"resistor.i", "resistor.r"}, "resistor.v")
    inward = graph.search({"resistor.v", "resistor.r"}, "resistor.i")

    assert outward is not None
    assert inward is not None

    assert [(step.operation, step.output, step.child_inputs) for step in outward] == [
        (sub_system, "resistor.v", frozenset({"i", "r"}))
    ]
    assert [(step.operation, step.output, step.child_inputs) for step in inward] == [
        (sub_system, "resistor.i", frozenset({"v", "r"}))
    ]


def test_search_returns_none_when_a_sub_system_port_is_unreachable_inside_the_child() -> None:
    graph = BipartiteGraph([_sealed_sub_system()], PYTHON_BACKEND)

    assert graph.search({"sealed.b"}, "sealed.a") is None


def test_search_terminates_on_mutually_derivable_sub_system_ports() -> None:
    sub_system = _resistor_sub_system()
    graph = BipartiteGraph([sub_system], PYTHON_BACKEND)

    assert graph.search({"resistor.i"}, "resistor.v") is None


def test_required_inputs_for_target_reports_the_inputs_a_plan_consumes() -> None:
    @transform
    def b(a: float) -> float:
        return a * 2

    @transform
    def c(b: float) -> float:
        return b + 1

    analyser = ExhaustiveStructuralAnalyser([b, c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target({"a"}, "c") == frozenset({"a"})


def test_required_inputs_for_target_is_none_when_no_route_exists() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target({"a"}, "c") is None


def test_required_inputs_for_target_is_none_for_an_unknown_output() -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = ExhaustiveStructuralAnalyser([c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target({"a", "b"}, "absent") is None


def test_required_inputs_for_target_ignores_what_the_plan_did_not_need() -> None:
    """A caller offering everything it could reach is told back only the part that gets used."""

    @transform
    def c(a: float) -> float:
        return a * 2

    analyser = ExhaustiveStructuralAnalyser([c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target({"a", "unused", "spare"}, "c") == frozenset({"a"})


def test_required_inputs_for_target_reaches_through_a_sub_system_step() -> None:
    """The step a sub-system search emits was never a graph node, so its inputs come off itself."""
    sub_system = _resistor_sub_system()
    analyser = ExhaustiveStructuralAnalyser([sub_system], PYTHON_BACKEND)

    required = analyser.required_inputs_for_target(
        {"resistor.i", "resistor.r"},
        "resistor.v",
    )

    assert required == frozenset({"resistor.i", "resistor.r"})


def test_realizing_a_sub_system_step_mangles_its_qualified_parameters() -> None:
    step = PlanStep(
        operation=_resistor_sub_system(),
        output="resistor.v",
        child_inputs=frozenset({"i", "r"}),
    )

    with EvaluationContext():
        realized = _operation_to_callable(
            step,
            PYTHON_BACKEND,
            symbol_domains={
                "resistor.i": SCALAR_DOMAIN,
                "resistor.r": SCALAR_DOMAIN,
            },
        )

    assert not isinstance(realized, RealizationFailure)
    assert realized.name == "resistor__v"
    assert realized.parameters == ("resistor__i", "resistor__r")
    assert realized.output_domain == SCALAR_DOMAIN


def _rig_over_sensor() -> Implementation:
    """A ``Rig`` whose three ports are each bridged to a ``Sensor`` sub-system.

    The sensor reaches ``reading`` two ways: through a ``scaled`` transform from ``raw``, and
    through an equation from ``calibration``. The first way is tried first, and it is the only
    one that goes through a transform named ``scaled``, so refusing that name leaves the second.
    """

    @system
    class Sensor(metaclass=Interface):
        reading: float = field()
        raw: float = field()
        calibration: float = field()

    @transform
    def scaled(raw: float) -> float:
        return raw * 2.0

    @equation
    def via_scaled(reading: float, scaled: float) -> bool:
        return reading == scaled

    @equation
    def calibrated(reading: float, calibration: float) -> bool:
        return reading == calibration + 1.0

    sensor = Implementation(
        Sensor,
        [scaled, via_scaled, calibrated],
    )

    @system
    class Rig(metaclass=Interface):
        answer: float = field()
        raw: float = field()
        calibration: float = field()

    @equation
    def answer_eq(answer: float, sensor: Sensor) -> bool:
        return answer == sensor.reading

    @equation
    def raw_eq(raw: float, sensor: Sensor) -> bool:
        return raw == sensor.raw

    @equation
    def calibration_eq(calibration: float, sensor: Sensor) -> bool:
        return calibration == sensor.calibration

    return Implementation(
        Rig,
        [answer_eq, raw_eq, calibration_eq],
        internal_implementations={"sensor": sensor},
    )


def test_a_sub_system_is_asked_again_after_the_way_it_first_named_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused way through a child must not take the child out of the parent's graph.

    The child prunes the refused step from its own graph, so asking it again names the way that
    is left. The parent settled on ``sensor.raw`` before the refusal was known, so
    only a second question can reach ``sensor.calibration``.
    """
    rig = _rig_over_sensor()
    expected_answer = 4.0
    attempts = _refuse_one_transform(monkeypatch, PYTHON_BACKEND, "scaled")

    case = _analyse_scalar(
        rig.analyser_for(PYTHON_BACKEND),
        {"raw", "calibration"},
        "answer",
    )

    assert attempts.count("scaled") == 1
    assert PYTHON_BACKEND.chain_case(case).callable(raw=1.0, calibration=3.0) == expected_answer


def test_retrying_a_sub_system_still_ends_when_the_backend_refuses_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each failure prunes a graph, so the retries run out instead of going on for ever."""
    rig = _rig_over_sensor()
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", _refuse_to_compile)

    with pytest.raises(BiltAnalysisError, match="Cannot compute answer"):
        _analyse_scalar(
            rig.analyser_for(PYTHON_BACKEND),
            {"raw", "calibration"},
            "answer",
        )


def test_removing_a_failed_sub_system_step_keeps_it_while_the_child_names_a_way() -> None:
    sub_system = _resistor_sub_system()
    graph = BipartiteGraph([sub_system], PYTHON_BACKEND)
    step = PlanStep(operation=sub_system, output="resistor.v")

    graph.remove_failed_plan_step(step)

    assert step in graph.graph


def test_removing_a_failed_sub_system_step_removes_it_once_the_child_names_none() -> None:
    sub_system = _sealed_sub_system()
    graph = BipartiteGraph([sub_system], PYTHON_BACKEND)
    step = PlanStep(operation=sub_system, output="sealed.a")

    graph.remove_failed_plan_step(step)

    assert step not in graph.graph
