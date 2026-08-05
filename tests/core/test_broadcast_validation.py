import pytest

from bilt.core._equation import equation
from bilt.core._structural_analysis._plan import (
    AnalysisPlan,
    PlanStep,
)
from bilt.core._tensors._broadcast_validation import (
    BroadcastValidationError,
    validate_broadcasts,
)
from bilt.core._tensors._tensors import tensor
from bilt.core._transform import transform
from bilt.symbolic._dimensions import dim
from bilt.symbolic._domains import Domain


@transform
def scaled(values: float, factor: float) -> float:
    return values * factor


@transform
def combined(scaled: float, times: float) -> float:
    return scaled + times


@equation
def relationship(
    left: float,
    right: float,
    result: float,
) -> bool:
    return result == left + right


def test_broadcasts_are_propagated_through_plan() -> None:
    location = dim("location")
    time = dim("time")

    plan = AnalysisPlan(
        inputs=frozenset({"values", "factor", "times"}),
        output="combined",
        steps=(
            PlanStep(
                operation=scaled,
                output="scaled",
            ),
            PlanStep(
                operation=combined,
                output="combined",
            ),
        ),
    )

    validated = validate_broadcasts(
        plan,
        {
            "values": tensor(
                [1.0, 2.0],
                over=location,
            ),
            "factor": 2.0,
            "times": tensor(
                [1.0, 2.0, 3.0],
                over=time,
            ),
        },
    )

    assert validated.steps[0].parameter_domains == (
        Domain((location,)),
        Domain.scalar(),
    )
    assert validated.steps[0].result_domain == Domain((location,))

    assert validated.steps[1].parameter_domains == (
        Domain((location,)),
        Domain((time,)),
    )
    assert validated.result_domain == Domain((location, time))


def test_broadcasts_are_propagated_through_isolated_equation() -> None:
    location = dim("location")
    time = dim("time")

    plan = AnalysisPlan(
        inputs=frozenset({"left", "right"}),
        output="result",
        steps=(
            PlanStep(
                operation=relationship,
                output="result",
            ),
        ),
    )

    validated = validate_broadcasts(
        plan,
        {
            "left": tensor(
                [1.0, 2.0],
                over=location,
            ),
            "right": tensor(
                [1.0, 2.0, 3.0],
                over=time,
            ),
        },
    )

    assert validated.steps[0].parameters == (
        "left",
        "right",
    )
    assert validated.steps[0].parameter_domains == (
        Domain((location,)),
        Domain((time,)),
    )
    assert validated.result_domain == Domain((location, time))


def test_unresolved_step_input_is_rejected() -> None:
    location = dim("location")

    plan = AnalysisPlan(
        inputs=frozenset({"values"}),
        output="combined",
        steps=(
            PlanStep(
                operation=combined,
                output="combined",
            ),
        ),
    )

    with pytest.raises(
        BroadcastValidationError,
        match="unresolved value 'scaled'",
    ):
        validate_broadcasts(
            plan,
            {
                "values": tensor(
                    [1.0, 2.0],
                    over=location,
                ),
            },
        )
