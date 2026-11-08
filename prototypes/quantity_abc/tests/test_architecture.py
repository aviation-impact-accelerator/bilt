"""Behavioural tests for the independent quantity/tensor architecture."""

from typing import Annotated, get_args, get_origin

import numpy as np
import pytest

from bilt.units import kilogram, meter, millimeter
from bilt.units.dimensions import MASS

from ..architecture._quantity import Quantity, ScalarQuantity
from ..architecture._quantity_tensor import QuantityTensor, typed
from ..architecture._tensor import Tensor, tensor
from ..architecture.typing_examples import scalar_model, tensor_model, total_mass


def test_scalar_construction_is_backwards_compatible() -> None:
    value = Quantity(2000.0, millimeter)
    converted = value.convert_to(meter)
    assert isinstance(converted, Quantity)
    assert converted.value == 2.0
    assert converted.unit is meter
    assert isinstance(2.0 * typed(kilogram), ScalarQuantity)


def test_tensor_quantity_is_both_types() -> None:
    mass = tensor([2.0, 3.0], over=("aircraft",)) * typed(kilogram)
    assert isinstance(mass, Quantity)
    assert isinstance(mass, Tensor)
    assert isinstance(mass, QuantityTensor)
    assert mass.magnitude.domain == ("aircraft",)
    assert np.array_equal(mass.magnitude.values, [2.0, 3.0])


def test_tensor_conversion_and_reduction() -> None:
    length = tensor([1000.0, 2000.0], over=("aircraft",)) * typed(millimeter)
    converted = length.convert_to(meter)
    assert isinstance(converted, QuantityTensor)
    assert np.array_equal(converted.values, [1.0, 2.0])
    assert converted.unit is meter
    total = total_mass(tensor([2.0, 3.0], over=("aircraft",)) * typed(kilogram), "aircraft")
    assert isinstance(total, Quantity)
    assert total.magnitude == 5.0


def test_shared_model_fields_accept_scalar_and_tensor_quantities() -> None:
    scalar = scalar_model()
    tensor_result = tensor_model()
    assert isinstance(scalar.doubled, Quantity)
    assert scalar.doubled.magnitude == 4.0
    assert isinstance(tensor_result.doubled, QuantityTensor)
    assert np.array_equal(tensor_result.doubled.values, [4.0, 6.0])


def test_annotated_magnitude_types_are_inspectable() -> None:
    scalar_or_tensor = Annotated[Quantity, MASS]
    tensor_only = Annotated[Quantity[Tensor], MASS]
    assert get_origin(scalar_or_tensor) is Annotated
    assert get_args(scalar_or_tensor)[1] is MASS
    assert get_origin(get_args(tensor_only)[0]) is Quantity


def test_mismatched_units_are_not_silently_added() -> None:
    with pytest.raises(Exception, match="same unit"):
        _ = Quantity(2.0, meter) + Quantity(3.0, millimeter)
