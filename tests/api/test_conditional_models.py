# ruff: noqa: SIM108
lazy from contextlib import AbstractContextManager

import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, dim, equation, field, system, tensor
from bilt.errors import BiltAnalysisError


@system
class Piecewise(metaclass=Interface):
    x: float = field()
    value: float = field()
    bias: float = field()
    total: float = field()


@equation(conditional=True)
def piecewise_value(x: float, value: float) -> bool:
    if x > 0:
        selected = x**2
    else:
        selected = -x
    return value == selected


@equation
def total_from_value(value: float, bias: float, total: float) -> bool:
    return total == value + bias


@system
class SelectedValue(metaclass=Interface):
    selector: float = field()
    threshold: float = field()
    positive: float = field()
    negative: float = field()
    result: float = field()


@equation(conditional=True)
def selected_value(selector: float, threshold: float, positive: float, negative: float, result: float) -> bool:
    if selector > threshold:
        selected = positive
    else:
        selected = negative
    return result == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(("x", "expected"), ((3.0, 9.0), (-3.0, 3.0), (0.0, 0.0)))
def test_piecewise_equation_through_public_implementation(
    backend: AbstractContextManager[object], x: float, expected: float
) -> None:
    implementation = Implementation(Piecewise, [piecewise_value])
    with backend, implementation:
        model = Piecewise(x=x)
        assert float(model.value) == pytest.approx(expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_conditional_equation_participates_in_multi_equation_plan(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Piecewise, [piecewise_value, total_from_value])
    with backend, implementation:
        positive = Piecewise(x=3.0, bias=10.0)
        negative = Piecewise(x=-2.0, bias=10.0)
        assert float(positive.total) == pytest.approx(19.0)
        assert float(negative.total) == pytest.approx(12.0)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
@pytest.mark.parametrize(("selector", "expected"), ((2.0, 6.0), (-2.0, 9.0)))
def test_implementation_tracks_inputs_from_both_conditional_branches(
    backend: AbstractContextManager[object], selector: float, expected: float
) -> None:
    implementation = Implementation(SelectedValue, [selected_value])
    with backend, implementation:
        model = SelectedValue(selector=selector, threshold=0.0, positive=6.0, negative=9.0)
        assert float(model.result) == pytest.approx(expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_missing_other_branch_input_has_no_supported_execution_plan(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(SelectedValue, [selected_value])
    with backend, implementation, pytest.raises(BiltAnalysisError):
        _ = SelectedValue(selector=2.0, threshold=0.0, positive=6.0).result


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_inverse_through_conditional_is_reported_as_analysis_failure(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Piecewise, [piecewise_value])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="Cannot compute x") as raised:
        _ = Piecewise(value=4.0).x
    assert "through a conditional" in str(raised.value)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_tensor_predicate_failure_is_reported_as_analysis_failure(backend: AbstractContextManager[object]) -> None:
    implementation = Implementation(Piecewise, [piecewise_value])
    with backend, implementation, pytest.raises(BiltAnalysisError, match="Cannot compute value") as raised:
        _ = Piecewise(x=tensor([1.0, -1.0], over=dim("site"))).value
    assert "scalar" in str(raised.value)
