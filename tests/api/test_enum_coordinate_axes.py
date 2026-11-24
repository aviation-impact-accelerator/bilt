"""Prototype categorical independent-variable axes without a selector tensor."""

import enum
import typing

import numpy as np
import pytest

from bilt import JAX_BACKEND, PYTHON_BACKEND, Implementation, Interface, Tensor, coordinate_axis, dim, equation, field, system
from bilt.core._domain_registry import DomainRegistry
from bilt.errors import BiltDomainRegistryError, BiltEvaluationError
from bilt.symbolic._dimensions import CoordinateAxis
from bilt.symbolic._errors import BiltInvalidDomainError

if typing.TYPE_CHECKING:
    from bilt._backend._base_backend import Backend


class Scenario(enum.Enum):
    MINIMUM = enum.auto()
    AVERAGE = enum.auto()
    MAXIMUM = enum.auto()


class Different(enum.IntEnum):
    MINIMUM = 1
    AVERAGE = 2
    MAXIMUM = 3


class Other(enum.IntEnum):
    MINIMUM = 1


@system
class Prices(metaclass=Interface):
    scenario: Scenario | CoordinateAxis = field()
    price: Tensor = field()


@equation
def price(scenario: Scenario | Tensor, price: float) -> bool:
    match scenario:
        case Scenario.MINIMUM:
            selected = 0.65
        case Scenario.AVERAGE:
            selected = 0.70
        case Scenario.MAXIMUM:
            selected = 0.75
    return price == selected


@pytest.mark.parametrize("backend", (PYTHON_BACKEND, JAX_BACKEND), ids=("python", "jax"))
def test_coordinate_axis_is_direct_independent_categorical_input(backend: Backend) -> None:
    scenario_dim = dim("scenario")
    axis = coordinate_axis(scenario_dim, tuple(Scenario))
    implementation = Implementation(Prices, [price])
    with backend, implementation:
        output = Prices(scenario=axis).price
    assert isinstance(output, Tensor)
    assert output.domain.dims == (scenario_dim,)
    np.testing.assert_allclose(output.values, [0.65, 0.70, 0.75])


def test_coordinate_axis_preserves_nominal_member_identity() -> None:
    scenario_dim = dim("scenario")
    axis = coordinate_axis(scenario_dim, tuple(Different))
    registry = DomainRegistry()
    registry.bind_axis(axis)
    assert registry.position(scenario_dim, Different.AVERAGE) == 1
    with pytest.raises(BiltDomainRegistryError, match="not present"):
        registry.position(scenario_dim, 2)
    with pytest.raises(BiltDomainRegistryError, match="not present"):
        registry.position(scenario_dim, Scenario.AVERAGE)


def test_categorical_axis_rejects_mixed_types_and_duplicates() -> None:
    scenario_dim = dim("scenario")
    with pytest.raises(BiltInvalidDomainError, match="one exact enum class"):
        coordinate_axis(scenario_dim, (Different.MINIMUM, Other.MINIMUM))
    with pytest.raises(BiltInvalidDomainError, match="cannot repeat"):
        coordinate_axis(scenario_dim, (Different.MINIMUM, Different.MINIMUM))


def test_wrong_enum_axis_rejected_at_interface_boundary() -> None:
    axis = coordinate_axis(dim("scenario"), tuple(Different))
    with pytest.raises(BiltEvaluationError, match="CoordinateAxis of Scenario members"):
        Prices(scenario=axis)


def test_numeric_axis_not_implicitly_categorical() -> None:
    with pytest.raises(BiltEvaluationError, match="CoordinateAxis of Scenario members"):
        Prices(scenario=coordinate_axis(dim("scenario"), (0, 1, 2)))
