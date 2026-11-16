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
from bilt.units import kilogram
from bilt.units.dimensions import MASS

Mass = typing.Annotated[Quantity, MASS]
PointwiseMass = typing.Annotated[Quantity | QuantityTensor, MASS]
time = dim("time")


@system
class DoubleMass(metaclass=Interface):
    mass: PointwiseMass = field()
    doubled: PointwiseMass = field()


@equation
def doubling(mass: Mass, doubled: Mass) -> bool:
    return doubled == mass * 2.0


def test_pointwise_quantity_alias_supports_scalar_and_tensor_values() -> None:
    with Implementation(DoubleMass, [doubling]):
        scalar = DoubleMass(mass=Quantity(2.0, kilogram)).doubled
        varied = DoubleMass(mass=tensor([2.0, 3.0], over=time, unit=kilogram)).doubled

    assert isinstance(scalar, Quantity)
    assert scalar.value == 2.0 * 2.0
    assert isinstance(varied, QuantityTensor)
    assert varied.unit == kilogram
    assert varied.domain.dims == (time,)
    np.testing.assert_allclose(varied.values, [4.0, 6.0])
