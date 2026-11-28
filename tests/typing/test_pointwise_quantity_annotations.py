"""Check pointwise scalar-or-tensor annotations in systems and equations."""

import typing

import numpy as np

from bilt import (
    Implementation,
    Interface,
    Quantity,
    QuantityTensor,
    dim,
    equation,
    field,
    system,
    tensor,
)
from bilt.units import UNITLESS, kilogram
from bilt.units.dimensions import DIMENSIONLESS, MASS

Mass = typing.Annotated[Quantity, MASS]
time = dim("time")


@system
class DoubleMass(metaclass=Interface):
    mass: Mass = field()
    doubled: Mass = field()


@equation
def doubling(mass: Mass, doubled: Mass) -> bool:
    return doubled == mass * 2.0


def test_pointwise_quantity_alias_supports_scalar_and_tensor_values() -> None:
    with Implementation(DoubleMass, [doubling]):
        scalar = DoubleMass(mass=2.0 * kilogram).doubled
        varied = DoubleMass(mass=tensor([2.0, 3.0], over=time, unit=kilogram)).doubled

    assert not isinstance(scalar, QuantityTensor)
    assert scalar.value == 2.0 * 2.0
    assert isinstance(varied, QuantityTensor)
    assert varied.unit == kilogram
    assert varied.domain.dims == (time,)
    np.testing.assert_allclose(varied.values, [4.0, 6.0])


Share = typing.Annotated[Quantity, DIMENSIONLESS]


@system
class Growth(metaclass=Interface):
    exponent: Share = field()
    result: Share = field()


@equation
def growth(exponent: Share, result: Share) -> bool:
    return result == 2.0**exponent


@equation
def bounded_growth(exponent: Share, result: Share) -> bool:
    if exponent < 1.0:  # noqa: SIM108 - symbolic conditional
        value = 2.0**exponent
    else:
        value = 2.0 * UNITLESS
    return result == value


def test_reflected_power_and_numeric_comparison_in_equations() -> None:
    with Implementation(Growth, [growth]):
        scalar = Growth(exponent=3.0 * UNITLESS).result
        varied = Growth(exponent=tensor([0.0, 1.0, 2.0], over=time, unit=UNITLESS)).result

    assert scalar.value == 2.0**3.0
    assert isinstance(varied, QuantityTensor)
    np.testing.assert_allclose(varied.values, [1.0, 2.0, 4.0])

    with Implementation(Growth, [bounded_growth]):
        bounded = Growth(exponent=tensor([0.0, 1.0, 2.0], over=time, unit=UNITLESS)).result

    assert isinstance(bounded, QuantityTensor)
    np.testing.assert_allclose(bounded.values, [1.0, 2.0, 2.0])
