import pytest

from bilt import Interface, field, system
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
from bilt.core._structural_analysis._realize import RealizationFailure, realize_plan

SUPPLY_VOLTAGE = 3.0


@system
class Resistor(metaclass=Interface):
    v: float = field()


@equation
def wiring(supply: float, resistor: Resistor) -> bool:
    return supply == resistor.v


def test_realize_plan_mangles_the_case_names() -> None:
    plan = AnalysisPlan(inputs=frozenset({"resistor.i", "resistor.r"}), output="resistor.v", steps=())

    case = realize_plan(plan, PYTHON_BACKEND)

    assert not isinstance(case, RealizationFailure)
    assert case.name == "resistor__v"
    assert case.return_name == "resistor__v"
    assert case.parameters == ("resistor__i", "resistor__r")


def test_realize_plan_leaves_plain_names_alone() -> None:
    plan = AnalysisPlan(inputs=frozenset({"a", "b"}), output="c", steps=())

    case = realize_plan(plan, PYTHON_BACKEND)

    assert not isinstance(case, RealizationFailure)
    assert case.name == "c"
    assert case.parameters == ("a", "b")


def test_realize_plan_rejects_an_unsupported_operation() -> None:
    class Unsupported:
        __bilt_name__: str = "unsupported"
        __bilt_parameters__: tuple[str, ...] = ("a",)

    plan = AnalysisPlan(inputs=frozenset(), output="c", steps=(PlanStep(operation=Unsupported(), output="c"),))

    with pytest.raises(NotImplementedError, match="is not supported for evaluation"):
        realize_plan(plan, PYTHON_BACKEND)


def test_realize_plan_mangles_an_equation_reading_a_dotted_port() -> None:
    plan = AnalysisPlan(
        inputs=frozenset({"resistor.v"}),
        output="supply",
        steps=(PlanStep(operation=wiring, output="supply"),),
    )

    case = realize_plan(plan, PYTHON_BACKEND)

    assert not isinstance(case, RealizationFailure)
    assert case.intermediates[0].name == "supply"
    assert case.intermediates[0].parameters == ("resistor__v",)
    assert case.intermediates[0].evaluate({"resistor__v": SUPPLY_VOLTAGE}) == SUPPLY_VOLTAGE


def test_realize_plan_mangles_an_equation_solving_for_a_dotted_port() -> None:
    plan = AnalysisPlan(
        inputs=frozenset({"supply"}),
        output="resistor.v",
        steps=(PlanStep(operation=wiring, output="resistor.v"),),
    )

    case = realize_plan(plan, PYTHON_BACKEND)

    assert not isinstance(case, RealizationFailure)
    assert case.intermediates[0].name == "resistor__v"
    assert case.intermediates[0].parameters == ("supply",)
    assert case.intermediates[0].evaluate({"supply": SUPPLY_VOLTAGE}) == SUPPLY_VOLTAGE
