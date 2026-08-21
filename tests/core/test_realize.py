import pytest

from bilt._backend._case import BackendCase
from bilt._backend._python._backend import PythonBackend
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._realize_analysis._realize import (
    RealizationFailure,
    case_to_callable,
    mangle_name,
    realize_plan,
)
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep

CURRENT = 2.0
RESISTANCE = 3.0


def _mangled_case(parameters: tuple[str, ...] = ("resistor__i", "resistor__r")) -> BackendCase:
    """A case spelled in mangled names, returning the first of them straight back."""
    return BackendCase(name="v", parameters=parameters, intermediates=[], return_name=parameters[0])


def test_mangle_name_leaves_a_plain_name_alone() -> None:
    assert mangle_name("voltage") == "voltage"


def test_mangle_name_replaces_the_dotted_separator() -> None:
    assert mangle_name("resistor.v") == "resistor__v"


def test_mangle_name_replaces_every_level_of_a_nested_name() -> None:
    assert mangle_name("board.resistor.v") == "board__resistor__v"


def test_case_to_callable_answers_in_the_names_the_request_asked_in() -> None:
    request = EvaluationRequest(output="resistor.v", available=frozenset({"resistor.i", "resistor.r"}))

    chained = case_to_callable(_mangled_case(), request, PythonBackend())

    assert chained.parameters == ("resistor.i", "resistor.r")


def test_case_to_callable_takes_its_parameter_order_from_the_case() -> None:
    request = EvaluationRequest(output="resistor.v", available=frozenset({"resistor.r", "resistor.i"}))

    chained = case_to_callable(_mangled_case(("resistor__r", "resistor__i")), request, PythonBackend())

    assert chained.parameters == ("resistor.r", "resistor.i")


def test_case_to_callable_rejects_a_name_the_case_does_not_have() -> None:
    request = EvaluationRequest(output="resistor.v", available=frozenset({"resistor.i", "resistor.typo"}))

    with pytest.raises(KeyError):
        case_to_callable(_mangled_case(), request, PythonBackend())


def test_the_callable_evaluates_under_its_canonical_names() -> None:
    request = EvaluationRequest(output="resistor.v", available=frozenset({"resistor.i", "resistor.r"}))

    chained = case_to_callable(_mangled_case(), request, PythonBackend())

    assert chained.evaluate({"resistor.i": CURRENT, "resistor.r": RESISTANCE}) == CURRENT


def test_realize_plan_mangles_the_case_names() -> None:
    plan = AnalysisPlan(inputs=frozenset({"resistor.i", "resistor.r"}), output="resistor.v", steps=())

    case = realize_plan(plan, PythonBackend())

    assert not isinstance(case, RealizationFailure)
    assert case.name == "resistor__v"
    assert case.return_name == "resistor__v"
    assert case.parameters == ("resistor__i", "resistor__r")


def test_realize_plan_leaves_plain_names_alone() -> None:
    plan = AnalysisPlan(inputs=frozenset({"a", "b"}), output="c", steps=())

    case = realize_plan(plan, PythonBackend())

    assert not isinstance(case, RealizationFailure)
    assert case.name == "c"
    assert case.parameters == ("a", "b")


def test_realize_plan_rejects_an_unsupported_operation() -> None:
    class Unsupported:
        __bilt_name__: str = "unsupported"
        __bilt_parameters__: tuple[str, ...] = ("a",)

    plan = AnalysisPlan(inputs=frozenset(), output="c", steps=(PlanStep(operation=Unsupported(), output="c"),))

    with pytest.raises(NotImplementedError, match="is not supported for evaluation"):
        realize_plan(plan, PythonBackend())
