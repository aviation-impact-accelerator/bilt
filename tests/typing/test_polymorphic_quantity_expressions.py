"""Experiment with one dimension alias for pointwise scalar-or-tensor quantities."""

import typing

import numpy as np

from bilt import (
    PYTHON_BACKEND,
    Implementation,
    Interface,
    Quantity,
    QuantityTensor,
    Tensor,
    dim,
    equation,
    field,
    system,
    tensor,
)
from bilt.units import kelvin, kilogram, watt
from bilt.units.dimensions import MASS, POWER, TEMPERATURE

Mass = typing.Annotated[Quantity | QuantityTensor, MASS]
TensorMass = typing.Annotated[QuantityTensor, MASS]
Temperature = typing.Annotated[Quantity | QuantityTensor, TEMPERATURE]
Heat = typing.Annotated[Quantity | QuantityTensor, POWER]
ThermalResistance = typing.Annotated[Quantity | QuantityTensor, TEMPERATURE / POWER]
time = dim("time")


@system
class DoubleMass(metaclass=Interface):
    mass: Mass = field()
    doubled: Mass = field()


@equation
def doubling(mass: Mass, doubled: Mass) -> bool:
    return doubled == mass * 2.0


@system
class ThermalPorts(metaclass=Interface):
    hot: Temperature = field()
    heat: Heat = field()


@system
class ThermalSource(metaclass=Interface):
    source: Temperature = field()
    resistance: ThermalResistance = field()


@equation
def thermal_supply(source: Temperature, resistance: ThermalResistance, inner: ThermalPorts) -> bool:
    return source == inner.hot + resistance * inner.heat


@system
class TensorTotals(metaclass=Interface):
    masses: TensorMass = field()
    total: Mass = field()


@equation
def sum_mass(masses: TensorMass, total: Mass) -> bool:
    return total == masses.sum(time)


def test_one_alias_accepts_scalar_and_tensor_inputs() -> None:
    with Implementation(DoubleMass, [doubling]):
        scalar = DoubleMass(mass=Quantity(2.0, kilogram)).doubled
        series = DoubleMass(mass=tensor([2.0, 3.0], over=time, unit=kilogram)).doubled

    assert isinstance(scalar, Quantity)
    assert scalar.unit == kilogram
    assert scalar.value == 2.0 * 2.0
    assert isinstance(series, QuantityTensor)
    assert series.unit == kilogram
    assert series.domain.dims == (time,)
    np.testing.assert_allclose(series.values, [4.0, 6.0])


def test_nested_pointwise_equation_has_no_typing_suppressions() -> None:
    child = Implementation(ThermalPorts, [])
    parent = Implementation(ThermalSource, [thermal_supply], internal_implementations={"inner": child})
    inputs: dict[str, object] = {
        "inner.hot": 10.0 * kelvin,
        "inner.heat": tensor([3.0, 4.0], over=time, unit=watt),
        "resistance": 2.0 * (kelvin / watt),
    }
    result = parent.evaluate(inputs, "source", PYTHON_BACKEND)
    assert isinstance(result, QuantityTensor)
    assert result.unit == kelvin
    assert result.domain.dims == (time,)
    np.testing.assert_allclose(result.values, [16.0, 18.0])


def test_tensor_specific_reduction_keeps_explicit_tensor_annotation() -> None:
    with Implementation(TensorTotals, [sum_mass]):
        result = TensorTotals(masses=tensor([2.0, 3.0], over=time, unit=kilogram)).total
    assert isinstance(result, Quantity)
    assert result.unit == kilogram
    assert result.value == 2.0 + 3.0


def _type_check_constructors(scalar: Quantity, series: QuantityTensor) -> None:
    """Check both concrete representations are valid inputs without casts."""
    _scalar_model = DoubleMass(mass=scalar)
    _tensor_model = DoubleMass(mass=series)
    if typing.TYPE_CHECKING:
        typing.assert_type(series.value, Tensor)
