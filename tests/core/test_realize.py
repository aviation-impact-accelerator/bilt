import pytest

from bilt._backend._python._backend import PythonBackend
from bilt.core._realize_analysis._realize import (
    RealizationFailure,
    mangle_inputs,
    realize_plan,
)
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep


def test_mangle_inputs_leaves_a_plain_name_alone() -> None:
    assert mangle_inputs({"voltage": 1.0}) == {"voltage": 1.0}


def test_mangle_inputs_replaces_the_dotted_separator() -> None:
    assert mangle_inputs({"resistor.v": 1.0}) == {"resistor__v": 1.0}


def test_mangle_inputs_replaces_every_level_of_a_nested_name() -> None:
    assert mangle_inputs({"board.resistor.v": 1.0}) == {"board__resistor__v": 1.0}


def test_mangle_inputs_carries_every_value_across() -> None:
    assert mangle_inputs({"resistor.i": 2.0, "r": 3.0}) == {"resistor__i": 2.0, "r": 3.0}


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
