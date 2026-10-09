"""Public behaviour of static selection in equations and transforms."""

import typing

import numpy as np
import pytest

from bilt import (
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Tensor,
    coordinate_axis,
    dim,
    domain,
    equation,
    field,
    system,
    tensor,
    transform,
)
from bilt.core._errors import BiltDomainBindingError
from bilt.core._evaluation_context import EvaluationContext
from bilt.core._tensors._errors import BiltTensorDomainError

if typing.TYPE_CHECKING:
    from bilt._backend import Backend

# Expected values describe observable selection behaviour.
# ruff: noqa: PLR2004


@pytest.fixture(params=("python", "jax"))
def backend(request: pytest.FixtureRequest) -> Backend:
    if request.param == "python":
        return PYTHON_BACKEND
    pytest.importorskip("jax")
    from bilt import JAX_BACKEND  # noqa: PLC0415

    return JAX_BACKEND


@pytest.mark.parametrize("consumer", ("equation", "transform"))
def test_static_selection_of_an_intermediate_reuses_the_current_coordinates(backend: Backend, consumer: str) -> None:
    time, location = dim("time"), dim("location")

    @system
    class Demo(metaclass=Interface):
        result: Tensor = field()

    @transform
    def scaled(values: Tensor) -> object:
        return values * 2

    @equation
    def select(scaled: Tensor, result: Tensor) -> bool:
        return result == scaled.sel(time, 2025)

    @transform
    def result(scaled: Tensor) -> object:
        """Selection can follow arithmetic and a local assignment."""
        adjusted = scaled + 10
        return adjusted.sel(time, 2025) - 10

    implementation = Implementation(Demo, [scaled, select if consumer == "equation" else result])
    for coordinates, expected in (
        ((2024, 2025, 2026), [4, 10]),
        ((2025, 2024, 2026), [2, 8]),
        ((2024, 2025, 2026), [4, 10]),
    ):
        with EvaluationContext() as evaluation:
            evaluation.domains.bind_axis(coordinate_axis(time, coordinates))
            output = implementation.evaluate(
                {"values": tensor([[1, 2, 3], [4, 5, 6]], over=(location, time))},
                "result",
                backend,
            )
        assert isinstance(output, Tensor)
        assert output.domain == domain(location)
        np.testing.assert_allclose(output.values, expected)


def test_transform_compound_selection_reduction_and_broadcasting(backend: Backend) -> None:
    time, location = dim("time"), dim("location")

    @system
    class Demo(metaclass=Interface):
        result: Tensor = field()

    @transform
    def result(values: Tensor, offset: Tensor) -> object:
        return (values * 2).sel(time, 2025).sum(location) + offset

    implementation = Implementation(Demo, [result])
    with EvaluationContext() as evaluation:
        evaluation.domains.bind_axis(coordinate_axis(time, (2024, 2025, 2026)))
        output = implementation.evaluate(
            {
                "values": tensor([[1, 2, 3], [4, 5, 6]], over=(location, time)),
                "offset": tensor([10, 20], over=location),
            },
            "result",
            backend,
        )
    assert isinstance(output, Tensor)
    assert output.domain == domain(location)
    np.testing.assert_allclose(output.values, [24, 34])


@pytest.mark.parametrize("consumer", ("equation", "transform"))
def test_positional_selection_accepts_a_scalar_expression(backend: Backend, consumer: str) -> None:
    time = dim("time")

    @system
    class Demo(metaclass=Interface):
        result: float = field()

    @equation
    def select(values: Tensor, index: int, result: float) -> bool:
        return result == values.isel(time, -index) * 2

    @transform
    def result(values: Tensor, index: int) -> object:
        return (values * 2).isel(time, -index)

    implementation = Implementation(Demo, [select if consumer == "equation" else result])
    assert implementation.evaluate({"values": tensor([10, 20, 30], over=time), "index": 1}, "result", backend) == 60


@pytest.mark.parametrize("consumer", ("equation", "transform"))
def test_missing_static_label_raises_a_domain_error(backend: Backend, consumer: str) -> None:
    time = dim("time")

    @system
    class Demo(metaclass=Interface):
        result: float = field()

    @equation
    def select(values: Tensor, result: float) -> bool:
        return result == values.sel(time, 2025)

    @transform
    def result(values: Tensor) -> object:
        return values.sel(time, 2025)

    implementation = Implementation(Demo, [select if consumer == "equation" else result])
    with pytest.raises(BiltDomainBindingError, match="not present"):
        implementation.evaluate(
            {"values": tensor([10, 20], over=time), "axis": coordinate_axis(time, (2024, 2026))}, "result", backend
        )


def test_selection_transform_rejects_control_flow_before_backend_execution(backend: Backend) -> None:
    time = dim("time")

    @system
    class Demo(metaclass=Interface):
        result: float = field()

    @transform
    def result(values: Tensor, choose: float) -> object:
        if choose:
            return values.sel(time, 2025)
        return 0

    implementation = Implementation(Demo, [result])
    with pytest.raises(BiltTensorDomainError, match="straight-line"):
        implementation.evaluate(
            {"values": tensor([10, 20], over=time), "axis": coordinate_axis(time, (2024, 2025)), "choose": 1},
            "result",
            backend,
        )


@pytest.mark.parametrize("consumer", ("equation", "transform"))
def test_positional_selector_must_be_scalar(backend: Backend, consumer: str) -> None:
    time = dim("time")

    @system
    class Demo(metaclass=Interface):
        result: float = field()

    @equation
    def select(values: Tensor, index: int, result: float) -> bool:
        return result == values.isel(time, index)

    @transform
    def result(values: Tensor, index: int) -> object:
        return values.isel(time, index)

    implementation = Implementation(Demo, [select if consumer == "equation" else result])
    with pytest.raises(BiltTensorDomainError, match="scalar selector"):
        # Deliberately pass a tensor to the scalar index parameter to check runtime validation.
        implementation.evaluate(
            {"values": tensor([10, 20], over=time), "index": tensor([0, 1], over=time)},
            "result",
            backend,
        )
