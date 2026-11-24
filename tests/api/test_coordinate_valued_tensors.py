"""Coordinate-valued tensors share the ordinary system and equation API."""

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
    coordinate_axis,
    dim,
    equation,
    field,
    system,
    tensor,
)
from bilt.core._domain_registry import DomainRegistry
from bilt.errors import BiltDomainRegistryError

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class Scenario(enum.IntEnum):
    MINIMUM = 1
    AVERAGE = 2
    MAXIMUM = 3


SCENARIO = dim("scenario")
ACCELERATION = dim("acceleration")
FORCE = dim("force")


@system
class Prices(metaclass=Interface):
    scenario: Scenario | Tensor = field()
    price: Tensor = field()


@equation
def price(scenario: Scenario | Tensor, price: float) -> bool:
    match scenario:
        case Scenario.MINIMUM:
            selected = 0.65
        case Scenario.AVERAGE:
            selected = 0.70
        case _:
            selected = 0.75
    return price == selected


@system
class Newton(metaclass=Interface):
    force: float | Tensor = field()
    mass: float = field()
    acceleration: float | Tensor = field()


@equation
def newtons_law(force: float, mass: float, acceleration: float) -> bool:
    return force == mass * acceleration


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_categorical_axis_is_an_ordinary_tensor(backend: Backend) -> None:
    axis = coordinate_axis(SCENARIO, (Scenario.MAXIMUM, Scenario.MINIMUM))
    selectors = tensor(axis)
    assert selectors.axes == (axis,)
    assert all(type(value) is Scenario for value in selectors.values)
    with backend, Implementation(Prices, [price]):
        result = Prices(scenario=selectors).price
    assert isinstance(result, Tensor)
    assert result.axes == (axis,)
    np.testing.assert_allclose(result.values, [0.75, 0.65])


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_choose_independent_variable_at_evaluation_time(backend: Backend) -> None:
    acceleration = coordinate_axis(ACCELERATION, (1.0, 2.0, 3.0))
    force = coordinate_axis(FORCE, (10.0, 20.0, 30.0))
    with backend, Implementation(Newton, [newtons_law]):
        forces = Newton(mass=10.0, acceleration=tensor(acceleration)).force
        accelerations = Newton(mass=10.0, force=tensor(force)).acceleration
    assert isinstance(forces, Tensor)
    assert isinstance(accelerations, Tensor)
    assert forces.axes == (acceleration,)
    assert accelerations.axes == (force,)
    np.testing.assert_allclose(forces.values, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(accelerations.values, [1.0, 2.0, 3.0])


def test_enum_coordinate_lookup_and_signature_are_nominal() -> None:
    registry = DomainRegistry()
    registry.bind(tensor(coordinate_axis(SCENARIO, tuple(Scenario))))
    assert registry.position(SCENARIO, Scenario.AVERAGE) == 1
    with pytest.raises(BiltDomainRegistryError, match="not present"):
        registry.position(SCENARIO, 2)
    numeric = DomainRegistry()
    numeric.bind(tensor(coordinate_axis(SCENARIO, (1, 2, 3))))
    assert registry.coordinate_signature() != numeric.coordinate_signature()
    with pytest.raises(BiltDomainRegistryError, match="conflicting coordinate"):
        registry.bind(tensor(coordinate_axis(SCENARIO, (1, 2, 3))))
