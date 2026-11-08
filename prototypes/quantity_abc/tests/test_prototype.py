"""Runtime checks for the experimental ABC and both concrete representations."""

from typing import Annotated, get_args, get_origin

import numpy as np
import pytest
from ..prototype import Quantity, QuantityTensor, ScalarQuantity

from bilt import Tensor, dim, tensor
from bilt.units import Quantity as NativeQuantity
from bilt.units import kilogram, meter, millimeter
from bilt.units.dimensions import LENGTH


def test_scalar_is_a_quantity_and_converts() -> None:
    length = ScalarQuantity(1000.0, millimeter)
    assert isinstance(length, Quantity)
    assert isinstance(length.magnitude, float)
    assert length.convert_to(meter).magnitude == pytest.approx(1.0)
    assert length.convert_to(meter).unit is meter


def test_tensor_is_both_a_tensor_and_a_quantity() -> None:
    time = dim("time")
    values = tensor([1000.0, 2000.0], over=time)
    length = QuantityTensor(values=values.values, domain=values.domain, unit=millimeter)
    assert isinstance(length, Quantity)
    assert isinstance(length, Tensor)
    assert length.unit is millimeter
    assert length.magnitude.unit is None
    assert np.array_equal(length.magnitude.values, [1000.0, 2000.0])
    converted = length.convert_to(meter)
    assert isinstance(converted, QuantityTensor)
    assert np.array_equal(converted.values, [1.0, 2.0])
    assert converted.domain == length.domain


def test_abstract_base_and_unitless_tensor() -> None:
    with pytest.raises(Exception, match="requires a unit"):
        QuantityTensor(values=np.array([1.0]), domain=tensor([1.0], over=dim("time")).domain, unit=None)


def test_annotated_preserves_dimension_metadata() -> None:
    length = Annotated[Quantity, LENGTH]
    assert get_origin(length) is Annotated
    assert get_args(length)[1] is LENGTH
    assert get_origin(get_args(Annotated[Quantity[Tensor], LENGTH])[0]) is Quantity


def test_existing_bilt_quantities_unchanged() -> None:
    assert isinstance(2.0 * kilogram, NativeQuantity)
    assert not isinstance(2.0 * kilogram, Quantity)


def test_system_implementation_example() -> None:
    from ..typing_examples import run_mass_scale

    assert run_mass_scale() == 4.0 * kilogram
