"""Tensor-valued enum match transforms use named elementwise selection."""

import enum
import typing

import numpy as np
import pytest

from bilt import (
    JAX_BACKEND,
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Tensor,
    dim,
    equation,
    field,
    system,
    tensor,
    transform,
)
from bilt.errors import BiltPortDeclarationError

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class Scenario(enum.Enum):
    MINIMUM = "minimum"
    AVERAGE = "average"
    MAXIMUM = "maximum"


class Fuel(enum.StrEnum):
    FOSSIL = "fossil"
    HYDROGEN = "hydrogen"


@system
class PriceModel(metaclass=Interface):
    scenario: Scenario | Tensor = field()
    price: Tensor = field()


@transform
def price(scenario: Scenario | Tensor) -> float:
    match scenario:
        case Scenario.MINIMUM:
            return 0.5
        case Scenario.AVERAGE:
            return 1.0
        case _:
            return 1.5


@system
class FuelModel(metaclass=Interface):
    scenario: Scenario | Tensor = field()
    fuel: Fuel | Tensor = field()


@transform
def fuel(scenario: Scenario | Tensor) -> Fuel | Tensor:
    match scenario:
        case Scenario.MINIMUM:
            return Fuel.FOSSIL
        case Scenario.AVERAGE:
            return Fuel.FOSSIL
        case _:
            return Fuel.HYDROGEN


@system
class ScaledPrices(metaclass=Interface):
    scenario: Scenario | Tensor = field()
    scale: Tensor = field()
    scaled_price: Tensor = field()


@transform
def scaled_price(scenario: Scenario | Tensor, scale: Tensor) -> Tensor:
    match scenario:
        case Scenario.MINIMUM:
            return 0.5 * scale
        case Scenario.AVERAGE:
            return scale
        case _:
            return 1.5 * scale


@system
class ParentModel(metaclass=Interface):
    total: Tensor = field()


@equation
def total(child: FuelModel, total: float) -> bool:
    match child.fuel:
        case Fuel.FOSSIL:
            selected = 10.0
        case _:
            selected = 20.0
    return total == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_transform_matches_enum_tensor_elementwise(backend: Backend) -> None:
    impl = Implementation(PriceModel, [price])
    selectors = tensor([Scenario.MAXIMUM, Scenario.MINIMUM, Scenario.AVERAGE], over=dim("scenario"))
    with backend, impl:
        output = PriceModel(scenario=selectors).price
    assert isinstance(output, Tensor)
    assert output.domain == selectors.domain
    np.testing.assert_allclose(output.values, [1.5, 0.5, 1.0])
    assert price(Scenario.MINIMUM) == pytest.approx(0.5)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_transform_returns_an_ordinary_enum_tensor(backend: Backend) -> None:
    impl = Implementation(FuelModel, [fuel])
    selectors = tensor([Scenario.MINIMUM, Scenario.MAXIMUM, Scenario.AVERAGE], over=dim("scenario"))
    with backend, impl:
        output = FuelModel(scenario=selectors).fuel
    assert isinstance(output, Tensor)
    assert output.domain == selectors.domain
    assert list(output.values) == [Fuel.FOSSIL, Fuel.HYDROGEN, Fuel.FOSSIL]
    assert all(type(value) is Fuel for value in output.values.flat)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_transform_broadcasts_enum_match_over_named_dimensions(backend: Backend) -> None:
    scenarios = tensor([Scenario.MINIMUM, Scenario.MAXIMUM], over=dim("scenario"))
    scales = tensor([10.0, 20.0, 30.0], over=dim("year"))
    impl = Implementation(ScaledPrices, [scaled_price])
    with backend, impl:
        output = ScaledPrices(scenario=scenarios, scale=scales).scaled_price
    assert isinstance(output, Tensor)
    assert set(output.domain.dims) == {dim("year"), dim("scenario")}
    expected = np.array([0.5, 1.5])[:, None] * np.array([10.0, 20.0, 30.0])[None, :]
    if output.domain.dims == (dim("year"), dim("scenario")):
        expected = expected.T
    np.testing.assert_allclose(output.values, expected)


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_child_transform_enum_tensor_flows_into_parent_equation(backend: Backend) -> None:
    child = Implementation(FuelModel, [fuel])
    parent = Implementation(ParentModel, [total], internal_implementations={"child": child})
    scenarios = tensor([Scenario.MINIMUM, Scenario.MAXIMUM, Scenario.AVERAGE], over=dim("scenario"))
    output = parent.evaluate({"child.scenario": scenarios}, "total", backend)
    assert isinstance(output, Tensor)
    assert output.domain == scenarios.domain
    np.testing.assert_allclose(output.values, [10.0, 20.0, 10.0])


def test_transform_rejects_other_enum_as_categorical_result() -> None:
    with pytest.raises(BiltPortDeclarationError, match="different enum class"):

        @transform
        def invalid(scenario: Scenario | Tensor) -> Fuel | Tensor:
            match scenario:
                case Scenario.MINIMUM:
                    return Scenario.MINIMUM  # pyrefly: ignore[bad-return]  # ty: ignore[invalid-return-type]
                case _:
                    return Fuel.FOSSIL
