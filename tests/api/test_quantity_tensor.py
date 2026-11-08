"""Unit-bearing tensors use a concrete class with precise factory types."""

import dataclasses
import typing

import numpy as np
import pytest

from bilt import Implementation, Interface, QuantityTensor, Tensor, dim, equation, field, system, tensor
from bilt.units import Quantity, Unit, absolute_celsius, absolute_kelvin, gram, kilogram, meter
from bilt.units._quantity_protocol import QuantityLike
from bilt.units.dimensions import MASS


def _unit_of(quantity: QuantityLike) -> Unit:
    return quantity.unit


def _tensor_magnitude(quantity: QuantityLike[Tensor]) -> Tensor:
    return quantity.value


def test_unit_bearing_factory_returns_quantity_tensor() -> None:
    time = dim("time")
    varied = tensor([1000.0, 2000.0], over=time, unit=gram)
    typing.assert_type(varied, QuantityTensor)
    typing.assert_type(varied.unit, Unit)
    typing.assert_type(varied.value, Tensor)

    assert isinstance(varied, QuantityTensor)
    assert isinstance(varied, Tensor)
    assert _unit_of(varied) is gram
    magnitude = _tensor_magnitude(varied)
    assert magnitude.unit is None
    assert magnitude.domain == varied.domain
    np.testing.assert_array_equal(magnitude.values, varied.values)


def test_unitless_factory_preserves_tensor_type() -> None:
    original = tensor([1.0, 2.0], over=dim("time"))
    typing.assert_type(original, Tensor)
    assert type(original) is Tensor
    assert original.unit is None


def test_optional_unit_factory_has_conservative_return_type() -> None:
    maybe_unit: Unit | None = gram
    result = tensor([1.0, 2.0], over=dim("time"), unit=maybe_unit)
    typing.assert_type(result, Tensor)
    assert isinstance(result, QuantityTensor)


def test_zero_dimensional_labelled_tensor_is_a_quantity_tensor() -> None:
    scalar = tensor(3.0, unit=kilogram)
    typing.assert_type(scalar, QuantityTensor)
    assert scalar.domain.is_scalar
    assert scalar.value.shape == ()
    assert scalar.unit is kilogram


def test_unit_labelling_and_composition_return_quantity_tensor() -> None:
    plain = tensor([1.0, 2.0], over=dim("time"))
    labelled = plain * gram
    typing.assert_type(labelled, QuantityTensor)
    assert labelled.unit is gram

    composed = labelled * meter
    typing.assert_type(composed, QuantityTensor)
    assert composed.unit == gram * meter

    divided = labelled / meter
    typing.assert_type(divided, QuantityTensor)
    assert divided.unit == gram / meter

    premultiplied = meter * labelled
    typing.assert_type(premultiplied, QuantityTensor)
    assert premultiplied.unit == meter * gram


def test_conversion_and_dataclasses_replace_preserve_quantity_tensor() -> None:
    original = tensor([1000.0, 2000.0], over=dim("time"), unit=gram)
    converted = original.convert_to(kilogram)
    typing.assert_type(converted, QuantityTensor)
    assert type(converted) is QuantityTensor
    assert converted.unit is kilogram
    np.testing.assert_array_equal(converted.values, [1.0, 2.0])

    replaced = dataclasses.replace(original, values=np.asarray([3.0, 4.0]))
    assert type(replaced) is QuantityTensor
    assert replaced.unit is gram


def test_absolute_conversion_and_identity_preserve_quantity_tensor() -> None:
    original = tensor([0.0, 100.0], over=dim("time"), unit=absolute_celsius)
    converted = original.convert_to(absolute_kelvin)
    assert isinstance(converted, QuantityTensor)
    np.testing.assert_allclose(converted.values, [273.15, 373.15])
    assert converted.convert_to(absolute_kelvin) is converted


def test_quantity_tensor_rejects_missing_unit() -> None:
    with pytest.raises(Exception, match="requires a physical unit"):
        QuantityTensor(values=np.asarray([1.0]), domain=tensor([1.0], over=dim("time")).domain, unit=None)  # type: ignore[arg-type]


type Mass = typing.Annotated[Quantity, MASS]


@system
class MassScale(metaclass=Interface):
    mass: Mass = field()
    doubled: Mass = field()


@equation
def double_mass(mass: Mass, doubled: Mass) -> bool:
    return doubled == mass * 2.0


def test_concrete_quantity_tensor_lifts_in_real_bilt_system() -> None:
    time = dim("time")
    original = tensor([1.0, 2.0, 3.0], over=time, unit=kilogram)
    with Implementation(MassScale, [double_mass]):
        result = MassScale(mass=original).doubled  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
    assert isinstance(result, Tensor)
    assert result.unit is kilogram
    np.testing.assert_allclose(result.values, [2.0, 4.0, 6.0])
