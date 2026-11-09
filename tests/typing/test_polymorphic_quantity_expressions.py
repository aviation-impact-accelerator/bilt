"""Experiment with one dimension alias for pointwise scalar-or-tensor quantities."""

import typing

import numpy as np
import pytest

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
from bilt.core._structural_analysis._linear_domains import equation_domain_hints
from bilt.symbolic._domains import Domain
from bilt.units import kelvin, kilogram, watt
from bilt.units.dimensions import MASS, POWER, TEMPERATURE

Mass = typing.Annotated[Quantity | QuantityTensor, MASS]
TensorMass = typing.Annotated[QuantityTensor, MASS]
HintedMass = typing.Annotated[Quantity | QuantityTensor, MASS, Domain.scalar()]
Temperature = typing.Annotated[Quantity | QuantityTensor, TEMPERATURE]
Heat = typing.Annotated[Quantity | QuantityTensor, POWER]
ThermalResistance = typing.Annotated[Quantity | QuantityTensor, TEMPERATURE / POWER]
time = dim("time")
RequiredTimeMass = typing.Annotated[QuantityTensor, MASS, Domain((time,))]


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


@equation
def hinted_sum_mass(masses: TensorMass, total: HintedMass) -> bool:
    return total == masses.sum(time)


@equation
def required_time_sum(masses: RequiredTimeMass, total: Mass) -> bool:
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


def test_exact_linear_hint_and_minimum_domain_requirement_are_distinct() -> None:
    """A pointwise hint stays exact; tensor requirements allow extra axes."""
    assert hinted_sum_mass.__bilt_port_declarations__["total"].required_domain is None
    assert equation_domain_hints(hinted_sum_mass) == (("total", Domain.scalar()),)
    assert required_time_sum.__bilt_port_declarations__["masses"].required_domain == Domain((time,))
    assert equation_domain_hints(required_time_sum) == ()

    scenario = dim("scenario")
    with Implementation(TensorTotals, [required_time_sum]):
        result = TensorTotals(masses=tensor([[2.0, 3.0], [4.0, 5.0]], over=(scenario, time), unit=kilogram)).total
    assert isinstance(result, QuantityTensor)
    assert result.domain.dims == (scenario,)
    np.testing.assert_allclose(result.values, [5.0, 9.0])


def test_union_arithmetic_is_not_an_eager_tensor_arithmetic_api() -> None:
    """Static types alone cannot detect that a symbolic-only tensor operator will fail eagerly."""
    scalar: Mass = Quantity(2.0, kilogram)
    series: Mass = tensor([2.0, 3.0], over=time, unit=kilogram)
    assert isinstance(scalar, Quantity)
    assert isinstance(series, QuantityTensor)
    assert (scalar * 2.0).value == 2.0 * 2.0
    with pytest.raises(NotImplementedError, match="Tensor arithmetic is not implemented"):
        _ = series * 2.0


def _eager_typing_limit(value: Mass) -> object:
    """Both checkers accept this expression, although concrete tensors reject it eagerly."""
    return value * 2.0
