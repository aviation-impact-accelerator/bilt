"""Check that tensor operators keep units at the type level.

Tensor arithmetic is symbolic: it raises outside a Bilt model, so these cases are
checked by pyrefly and ty over this module rather than run. The one runtime test
confirms that the overrides on `QuantityTensor` still label with units.
"""

import typing

import numpy as np

from bilt import Quantity, QuantityTensor, Tensor, dim, tensor
from bilt.units import kilogram, meter

time = dim("time")


def _quantity_tensor_arithmetic_keeps_its_unit(qt: QuantityTensor, t: Tensor) -> None:
    typing.assert_type(qt + qt, QuantityTensor)
    typing.assert_type(qt - qt, QuantityTensor)
    typing.assert_type(-qt, QuantityTensor)
    typing.assert_type(+qt, QuantityTensor)
    typing.assert_type(abs(qt), QuantityTensor)
    typing.assert_type(qt * 2.0, QuantityTensor)
    typing.assert_type(2.0 * qt, QuantityTensor)
    typing.assert_type(qt * 2, QuantityTensor)
    typing.assert_type(qt / 2.0, QuantityTensor)
    typing.assert_type(2.0 / qt, QuantityTensor)
    typing.assert_type(qt * qt, QuantityTensor)
    typing.assert_type(qt / qt, QuantityTensor)
    typing.assert_type(qt * t, QuantityTensor)
    typing.assert_type(t * qt, QuantityTensor)
    typing.assert_type(qt / t, QuantityTensor)
    typing.assert_type(t / qt, QuantityTensor)
    typing.assert_type(qt**2, QuantityTensor)
    typing.assert_type(qt**0.5, QuantityTensor)


def _mixed_scalar_and_tensor_quantities_give_a_tensor(q: Quantity, qt: QuantityTensor, t: Tensor) -> None:
    typing.assert_type(q + qt, QuantityTensor)
    typing.assert_type(qt + q, QuantityTensor)
    typing.assert_type(q - qt, QuantityTensor)
    typing.assert_type(qt - q, QuantityTensor)
    typing.assert_type(q * qt, QuantityTensor)
    typing.assert_type(qt * q, QuantityTensor)
    typing.assert_type(q / qt, QuantityTensor)
    typing.assert_type(qt / q, QuantityTensor)
    typing.assert_type(q * t, QuantityTensor)
    typing.assert_type(t * q, QuantityTensor)
    typing.assert_type(t / q, QuantityTensor)
    typing.assert_type(q / t, QuantityTensor)


def _unitless_tensor_arithmetic_stays_unitless(t: Tensor) -> None:
    typing.assert_type(t + t, Tensor)
    typing.assert_type(t * 2.0, Tensor)
    typing.assert_type(2.0 / t, Tensor)
    typing.assert_type(-t, Tensor)
    typing.assert_type(t**2, Tensor)
    typing.assert_type(2.0**t, Tensor)
    typing.assert_type(t * meter, QuantityTensor)


def _reductions_and_selections_keep_the_carrier(qt: QuantityTensor, t: Tensor) -> None:
    typing.assert_type(qt.sum(time), QuantityTensor)
    typing.assert_type(qt.mean(time), QuantityTensor)
    typing.assert_type(qt.min(time), QuantityTensor)
    typing.assert_type(qt.max(time), QuantityTensor)
    typing.assert_type(qt.prod(time), QuantityTensor)
    typing.assert_type(qt.isel(time, 0), QuantityTensor)
    typing.assert_type(t.sum(time), Tensor)
    typing.assert_type(qt.value, Tensor)


def _union_arithmetic_type_checks(u: Quantity | QuantityTensor) -> None:
    typing.assert_type(u + u, Quantity | QuantityTensor)
    typing.assert_type(u - u, Quantity | QuantityTensor)
    typing.assert_type(u * u, Quantity | QuantityTensor)
    typing.assert_type(u / u, Quantity | QuantityTensor)
    typing.assert_type(u * 2.0, Quantity | QuantityTensor)
    typing.assert_type(u**2, Quantity | QuantityTensor)


def test_quantity_tensor_overrides_still_label_with_units() -> None:
    labelled = tensor([1.0, 2.0], over=time, unit=kilogram)

    for result, unit in (
        (labelled * meter, kilogram * meter),
        (meter * labelled, meter * kilogram),
        (labelled / meter, kilogram / meter),
    ):
        assert isinstance(result, QuantityTensor)
        assert result.unit == unit
        np.testing.assert_array_equal(result.values, [1.0, 2.0])
