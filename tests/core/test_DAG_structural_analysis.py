import typing

import pytest

from bilt import Implementation, Interface, field, system, transform
from bilt._backend._jax._backend import JAX_BACKEND
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis import StructuralAnalyser
from bilt.core._structural_analysis._causalization_block_solver._backend import (
    CausalizationStructuralAnalyzer,
)
from bilt.core._structural_analysis._causalization_block_solver._errors import (
    BiltOverconstrainedError,
)
from bilt.core._sub_system import SubSystemConstituent
from bilt.errors import BiltAnalysisError, BiltIsolationError, BiltUnsupportedNodeError
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    import ast

    from bilt._backend._base_backend import Backend
    from bilt._backend._callable import BackendCallable
    from bilt._backend._case import BackendCase
    from bilt.core._model_constituent import BiltModelConstituent

type AnalyserFactory = typing.Callable[
    [typing.Collection[BiltModelConstituent], Backend],
    StructuralAnalyser,
]


@pytest.fixture(autouse=True)
def _select_structural_analyser(analyser_factory: object) -> None:
    """Select the analyser before constructing test implementations."""
    _ = analyser_factory


@typing.runtime_checkable
class _RuntimeStructuralAnalyser(StructuralAnalyser, typing.Protocol):
    """Runtime-checkable view of the structural analyser protocol."""


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
    msg = "AST node 'While' is unsupported by this backend."
    raise BiltUnsupportedNodeError(msg)


def _refuse_one_transform(
    monkeypatch: pytest.MonkeyPatch,
    backend: Backend,
    refused: str,
) -> list[str]:
    attempts: list[str] = []
    transpile = backend.transpile

    def counting_transpile(
        function_definition: ast.FunctionDef,
    ) -> BackendCallable:
        attempts.append(function_definition.name)
        if function_definition.name == refused:
            _refuse_to_compile(function_definition)
        return transpile(function_definition)

    monkeypatch.setattr(backend, "transpile", counting_transpile)
    return attempts


def test_analyser_conforms_to_the_structural_analyser_protocol(
    analyser_factory: AnalyserFactory,
) -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = analyser_factory([c], PYTHON_BACKEND)

    assert isinstance(analyser, _RuntimeStructuralAnalyser)


def test_analyser_produces_a_working_callable(
    analyser_factory: AnalyserFactory,
) -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    expected_c = 3.0
    result = _analyse_scalar(
        analyser_factory([c], PYTHON_BACKEND),
        {"a", "b"},
        "c",
    )
    backend_case = PYTHON_BACKEND.chain_case(result)

    assert backend_case.callable(a=1.0, b=2.0) == expected_c
    assert result.intermediates[-1].output_domain == SCALAR_DOMAIN


def test_analyser_finds_a_realisable_route_across_repeated_queries(
    analyser_factory: AnalyserFactory,
) -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def linear(x: float, w: float) -> bool:
        return x == w + 1.0

    expected_x = 5.0
    analyser = analyser_factory([square, linear], PYTHON_BACKEND)

    if isinstance(analyser, CausalizationStructuralAnalyzer):
        for _ in range(2):
            with pytest.raises(BiltOverconstrainedError):
                _analyse_scalar(analyser, {"z", "w"}, "x")
        return

    for _ in range(2):
        case = PYTHON_BACKEND.chain_case(
            _analyse_scalar(analyser, {"z", "w"}, "x"),
        )
        assert case.callable(w=4.0, z=25.0) == expected_x


def test_analyser_failure_leaves_other_equation_directions_usable(
    analyser_factory: AnalyserFactory,
) -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    @equation
    def offset(z: float, y: float) -> bool:
        return z == y + 1.0

    expected_z = 5.0
    analyser = analyser_factory([square, offset], PYTHON_BACKEND)

    with pytest.raises(BiltAnalysisError, match="Cannot compute x"):
        _analyse_scalar(analyser, {"z"}, "x")

    case = PYTHON_BACKEND.chain_case(
        _analyse_scalar(analyser, {"y"}, "z"),
    )

    assert case.callable(y=4.0) == expected_z


def test_a_refused_step_is_only_attempted_once_across_queries(
    analyser_factory: AnalyserFactory,
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

    analyser = analyser_factory([b, c, d], PYTHON_BACKEND)
    attempts = _refuse_one_transform(monkeypatch, PYTHON_BACKEND, "b")

    for output in ("c", "d"):
        with pytest.raises(BiltAnalysisError):
            _analyse_scalar(analyser, {"a"}, output)

    assert attempts == ["b"]


def test_a_step_refused_by_one_backend_remains_available_to_another(
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


def test_exhausted_search_reports_the_backend_failure(
    analyser_factory: AnalyserFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @equation
    def from_w(x: float, w: float) -> bool:
        return x == w + 1.0

    analyser = analyser_factory([from_w], PYTHON_BACKEND)
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", _refuse_to_compile)

    with pytest.raises(BiltAnalysisError, match="unsupported") as raised:
        _analyse_scalar(analyser, {"w"}, "x")

    assert isinstance(raised.value.__cause__, BiltUnsupportedNodeError)


def test_exhausted_search_reports_the_isolation_failure(
    analyser_factory: AnalyserFactory,
) -> None:
    @equation
    def square(z: float, x: float) -> bool:
        return z == x * x

    analyser = analyser_factory([square], PYTHON_BACKEND)

    with pytest.raises(BiltAnalysisError) as raised:
        _analyse_scalar(analyser, {"z"}, "x")

    assert isinstance(raised.value.__cause__, BiltIsolationError)


def _resistor_sub_system() -> SubSystemConstituent:
    @system
    class Resistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    return SubSystemConstituent(
        name="resistor",
        implementation=Implementation(Resistor, [ohms_law]),
    )


def test_analyser_solves_a_subsystem_in_either_direction(
    analyser_factory: AnalyserFactory,
) -> None:
    expected_voltage = 6.0
    expected_current = 2.0
    analyser = analyser_factory(
        [_resistor_sub_system()],
        PYTHON_BACKEND,
    )

    outward = PYTHON_BACKEND.chain_case(
        _analyse_scalar(
            analyser,
            {"resistor.i", "resistor.r"},
            "resistor.v",
        ),
    )
    inward = PYTHON_BACKEND.chain_case(
        _analyse_scalar(
            analyser,
            {"resistor.v", "resistor.r"},
            "resistor.i",
        ),
    )

    assert outward.callable(resistor__i=2.0, resistor__r=3.0) == expected_voltage
    assert inward.callable(resistor__v=6.0, resistor__r=3.0) == expected_current


def test_required_inputs_reports_consumed_inputs(
    analyser_factory: AnalyserFactory,
) -> None:
    @transform
    def b(a: float) -> float:
        return a * 2

    @transform
    def c(b: float) -> float:
        return b + 1

    analyser = analyser_factory([b, c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target(
        {"a"},
        "c",
    ) == frozenset({"a"})


def test_required_inputs_is_none_when_no_route_exists(
    analyser_factory: AnalyserFactory,
) -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = analyser_factory([c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target({"a"}, "c") is None


def test_required_inputs_is_none_for_an_unknown_output(
    analyser_factory: AnalyserFactory,
) -> None:
    @transform
    def c(a: float, b: float) -> float:
        return a + b

    analyser = analyser_factory([c], PYTHON_BACKEND)

    assert (
        analyser.required_inputs_for_target(
            {"a", "b"},
            "absent",
        )
        is None
    )


def test_required_inputs_ignores_unused_inputs(
    analyser_factory: AnalyserFactory,
) -> None:
    @transform
    def c(a: float) -> float:
        return a * 2

    analyser = analyser_factory([c], PYTHON_BACKEND)

    assert analyser.required_inputs_for_target(
        {"a", "unused", "spare"},
        "c",
    ) == frozenset({"a"})


def test_required_inputs_reaches_through_a_subsystem(
    analyser_factory: AnalyserFactory,
) -> None:
    analyser = analyser_factory(
        [_resistor_sub_system()],
        PYTHON_BACKEND,
    )

    assert analyser.required_inputs_for_target(
        {"resistor.i", "resistor.r"},
        "resistor.v",
    ) == frozenset({"resistor.i", "resistor.r"})


def _rig_over_sensor() -> Implementation:
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


def test_parent_child_alternatives_follow_the_structural_constraint_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_answer = 4.0
    rig = _rig_over_sensor()
    attempts = _refuse_one_transform(monkeypatch, PYTHON_BACKEND, "scaled")
    analyser = rig.analyser_for(PYTHON_BACKEND)
    if isinstance(analyser, CausalizationStructuralAnalyzer):
        with pytest.raises(BiltOverconstrainedError, match=r"sensor\.calibrated"):
            _analyse_scalar(analyser, {"raw", "calibration"}, "answer")
        assert attempts == []
        return

    case = _analyse_scalar(
        analyser,
        {"raw", "calibration"},
        "answer",
    )
    result = PYTHON_BACKEND.chain_case(case)

    assert result.callable(raw=1.0, calibration=3.0) == expected_answer


def test_subsystem_retry_terminates_when_the_backend_refuses_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Supply only raw: calibration is now another unknown constrained by the
    # child, rather than a competing independently prescribed boundary value.
    rig = _rig_over_sensor()
    monkeypatch.setattr(PYTHON_BACKEND, "transpile", _refuse_to_compile)

    with pytest.raises(BiltAnalysisError, match="Cannot compute answer"):
        _analyse_scalar(rig.analyser_for(PYTHON_BACKEND), {"raw"}, "answer")
