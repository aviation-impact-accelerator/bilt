import typing

import pytest

from bilt import Implementation, Interface, field, system
from bilt._backend._python._backend import PYTHON_BACKEND
from bilt.core._equation import equation
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._evaluation_request import EvaluationRequest
from bilt.core._structural_analysis._plan import AnalysisPlan, PlanStep
from bilt.core._structural_analysis._realize import RealizationFailure, realize_plan
from bilt.core._sub_system import SubSystemConstituent
from bilt.symbolic._domains import Domain

if typing.TYPE_CHECKING:
    from bilt.core._port_declaration import PortDeclaration

SUPPLY_VOLTAGE = 3.0
SCALAR_DOMAIN = Domain.scalar()


@system
class Resistor(metaclass=Interface):
    v: float = field()


@equation
def wiring(supply: float, resistor: Resistor) -> bool:
    return supply == resistor.v


def _scalar_plan(
    *,
    inputs: frozenset[str],
    output: str,
    steps: tuple[PlanStep, ...],
) -> AnalysisPlan:
    return AnalysisPlan(
        request=EvaluationRequest(
            output=output,
            input_domains=dict.fromkeys(inputs, SCALAR_DOMAIN),
        ),
        steps=steps,
    )


def _realize_scalar_plan(plan: AnalysisPlan):  # noqa: ANN202
    with EvaluationContext():
        return realize_plan(
            plan,
            PYTHON_BACKEND,
        )


def test_realize_plan_mangles_the_case_names() -> None:
    plan = _scalar_plan(
        inputs=frozenset({"resistor.i", "resistor.r"}),
        output="resistor.v",
        steps=(),
    )

    case = _realize_scalar_plan(plan)

    assert not isinstance(case, RealizationFailure)
    assert case.name == "resistor__v"
    assert case.return_name == "resistor__v"
    assert case.parameters == ("resistor__i", "resistor__r")


def test_realize_plan_leaves_plain_names_alone() -> None:
    plan = _scalar_plan(
        inputs=frozenset({"a", "b"}),
        output="c",
        steps=(),
    )

    case = _realize_scalar_plan(plan)

    assert not isinstance(case, RealizationFailure)
    assert case.name == "c"
    assert case.parameters == ("a", "b")


def test_realize_plan_rejects_an_unsupported_operation() -> None:
    class Unsupported:
        __bilt_name__: str = "unsupported"
        __bilt_parameters__: tuple[str, ...] = ("a",)
        __bilt_port_declarations__: dict[str, PortDeclaration] = {}  # noqa: RUF012

    plan = _scalar_plan(
        inputs=frozenset(),
        output="c",
        steps=(
            PlanStep(
                operation=Unsupported(),
                output="c",
            ),
        ),
    )

    with pytest.raises(NotImplementedError, match="is not supported for evaluation"):
        _realize_scalar_plan(plan)


def test_realize_plan_mangles_an_equation_reading_a_dotted_port() -> None:
    plan = _scalar_plan(
        inputs=frozenset({"resistor.v"}),
        output="supply",
        steps=(
            PlanStep(
                operation=wiring,
                output="supply",
            ),
        ),
    )

    case = _realize_scalar_plan(plan)

    assert not isinstance(case, RealizationFailure)
    assert case.name == "supply"
    assert case.parameters == ("resistor__v",)
    assert case.intermediates[0].name == "supply"
    assert case.intermediates[0].parameters == ("resistor__v",)
    assert case.intermediates[0].evaluate({"resistor__v": SUPPLY_VOLTAGE}) == SUPPLY_VOLTAGE
    assert case.intermediates[0].output_domain == SCALAR_DOMAIN


def test_realize_plan_mangles_an_equation_solving_for_a_dotted_port() -> None:
    plan = _scalar_plan(
        inputs=frozenset({"supply"}),
        output="resistor.v",
        steps=(
            PlanStep(
                operation=wiring,
                output="resistor.v",
            ),
        ),
    )

    case = _realize_scalar_plan(plan)

    assert not isinstance(case, RealizationFailure)
    assert case.name == "resistor__v"
    assert case.parameters == ("supply",)
    assert case.intermediates[0].name == "resistor__v"
    assert case.intermediates[0].parameters == ("supply",)
    assert case.intermediates[0].evaluate({"supply": SUPPLY_VOLTAGE}) == SUPPLY_VOLTAGE
    assert case.intermediates[0].output_domain == SCALAR_DOMAIN


def test_realize_plan_accepts_a_transform_decorated_through_the_package() -> None:
    import bilt  # noqa: PLC0415

    @bilt.transform
    def doubled(a: float) -> float:
        return 2 * a

    plan = _scalar_plan(
        inputs=frozenset({"a"}),
        output="c",
        steps=(
            PlanStep(
                operation=doubled,
                output="c",
            ),
        ),
    )

    case = _realize_scalar_plan(plan)

    assert not isinstance(case, RealizationFailure)
    assert case.intermediates[0].evaluate({"a": SUPPLY_VOLTAGE}) == 2 * SUPPLY_VOLTAGE
    assert case.intermediates[0].output_domain == SCALAR_DOMAIN


def test_realize_plan_accepts_a_transform_decorated_through_an_alias() -> None:
    from bilt import transform as renamed_transform  # noqa: PLC0415

    @renamed_transform
    def doubled(a: float) -> float:
        return 2 * a

    plan = _scalar_plan(
        inputs=frozenset({"a"}),
        output="c",
        steps=(
            PlanStep(
                operation=doubled,
                output="c",
            ),
        ),
    )

    case = _realize_scalar_plan(plan)

    assert not isinstance(case, RealizationFailure)
    assert case.intermediates[0].evaluate({"a": SUPPLY_VOLTAGE}) == 2 * SUPPLY_VOLTAGE
    assert case.intermediates[0].output_domain == SCALAR_DOMAIN


def test_realize_plan_mangles_subsystem_parameters() -> None:
    @system
    class ChildResistor(metaclass=Interface):
        v: float = field()
        i: float = field()
        r: float = field()

    @equation
    def ohms_law(v: float, i: float, r: float) -> bool:
        return v == i * r

    sub_system = SubSystemConstituent(
        name="resistor",
        implementation=Implementation(ChildResistor, [ohms_law]),
    )
    plan = _scalar_plan(
        inputs=frozenset({"resistor.i", "resistor.r"}),
        output="resistor.v",
        steps=(
            PlanStep(
                operation=sub_system,
                output="resistor.v",
                child_inputs=frozenset({"i", "r"}),
            ),
        ),
    )

    case = _realize_scalar_plan(plan)

    result = 6.0

    assert not isinstance(case, RealizationFailure)
    assert case.name == "resistor__v"
    assert case.parameters == ("resistor__i", "resistor__r")
    assert case.intermediates[0].name == "resistor__v"
    assert case.intermediates[0].parameters == ("resistor__i", "resistor__r")
    assert case.intermediates[0].output_domain == SCALAR_DOMAIN
    assert (
        case.intermediates[0].evaluate(
            {"resistor__i": 2.0, "resistor__r": 3.0},
        )
        == result
    )
