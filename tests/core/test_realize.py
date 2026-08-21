import pytest

from bilt._backend._python._backend import PythonBackend
from bilt.core._realize_analysis._realize import RealizationFailure, realize_plan
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep


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
