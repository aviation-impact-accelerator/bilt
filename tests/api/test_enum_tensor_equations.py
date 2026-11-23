"""Ordinary enum-valued tensors select equation branches elementwise."""

import enum
from contextlib import AbstractContextManager

import numpy as np
import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, Tensor, dim, equation, field, system, tensor
from bilt.errors import BiltEvaluationError
from bilt.symbolic._domains import domain


class Scenario(enum.Enum):
    MINIMUM = "minimum"
    AVERAGE = "average"
    MAXIMUM = "maximum"


@system
class ScenarioCalculation(metaclass=Interface):
    scenario: Scenario | Tensor = field()
    multiplier: Tensor = field()
    result: Tensor = field()


@equation
def result(scenario: Scenario | Tensor, multiplier: float, result: float) -> bool:
    match scenario:
        case Scenario.MINIMUM:
            selected = 0.5
        case Scenario.AVERAGE:
            selected = 1.0
        case _:
            selected = 1.5
    return result == selected * multiplier


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_enum_tensor_match_is_elementwise(backend: AbstractContextManager[object]) -> None:
    scenario_axis = dim("scenario")
    implementation = Implementation(ScenarioCalculation, [result])
    values = tensor([Scenario.MINIMUM, Scenario.AVERAGE, Scenario.MAXIMUM], over=scenario_axis)
    with backend, implementation:
        output = ScenarioCalculation(scenario=values, multiplier=tensor([8.0, 8.0, 8.0], over=scenario_axis)).result
    assert isinstance(output, Tensor)
    assert output.domain == domain(scenario_axis)
    np.testing.assert_allclose(output.values, [4.0, 8.0, 12.0])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_enum_selector_broadcasts_over_other_named_dimensions(backend: AbstractContextManager[object]) -> None:
    scenario_axis = dim("scenario")
    time_axis = dim("time")
    implementation = Implementation(ScenarioCalculation, [result])
    scenarios = tensor([Scenario.MINIMUM, Scenario.MAXIMUM], over=scenario_axis)
    multipliers = tensor([10.0, 20.0, 30.0], over=time_axis)
    with backend, implementation:
        output = ScenarioCalculation(scenario=scenarios, multiplier=multipliers).result
    assert isinstance(output, Tensor)
    assert set(output.domain.dims) == {scenario_axis, time_axis}
    expected = np.array([0.5, 1.5])[:, None] * np.array([10.0, 20.0, 30.0])[None, :]
    if output.domain.dims == (time_axis, scenario_axis):
        expected = expected.T
    np.testing.assert_allclose(output.values, expected)


def test_invalid_enum_tensor_is_rejected_before_backend_execution() -> None:
    impl = Implementation(ScenarioCalculation, [result])
    bad = tensor([0, 1, 2], over=dim("scenario"))
    with pytest.raises(BiltEvaluationError, match="requires a Tensor of Scenario members"):
        impl.evaluate({"scenario": bad, "multiplier": 2.0}, "result", PYTHON_BACKEND)


def test_enum_input_is_preserved_when_returned_directly() -> None:
    scenarios = tensor([Scenario.MINIMUM, Scenario.MAXIMUM], over=dim("scenario"))
    implementation = Implementation(ScenarioCalculation, [result])
    assert implementation.evaluate({"scenario": scenarios}, "scenario", PYTHON_BACKEND) is scenarios
