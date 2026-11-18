"""Check that the concrete tensor classes' operators keep units at the type level.

These are the private classes behind the public protocols (see
`test_quantity_protocols.py`). Tensor arithmetic is symbolic: it raises outside a
Bilt model, so these cases are checked by pyrefly and ty over this module rather
than run. The one runtime test confirms that the overrides on the concrete quantity
tensor still label with units.
"""

import typing

import numpy as np

from bilt import QuantityTensor, dim, tensor
from bilt.core._tensors._tensors import _QuantityTensor, _Tensor
from bilt.units import kilogram, meter
from bilt.units._quantities import _Quantity

time = dim("time")


def _quantity_tensor_arithmetic_keeps_its_unit(qt: _QuantityTensor, t: _Tensor) -> None:
    typing.assert_type(qt + qt, _QuantityTensor)
    typing.assert_type(qt - qt, _QuantityTensor)
    typing.assert_type(-qt, _QuantityTensor)
    typing.assert_type(+qt, _QuantityTensor)
    typing.assert_type(abs(qt), _QuantityTensor)
    typing.assert_type(qt * 2.0, _QuantityTensor)
    typing.assert_type(2.0 * qt, _QuantityTensor)
    typing.assert_type(qt * 2, _QuantityTensor)
    typing.assert_type(qt / 2.0, _QuantityTensor)
    typing.assert_type(2.0 / qt, _QuantityTensor)
    typing.assert_type(qt * qt, _QuantityTensor)
    typing.assert_type(qt / qt, _QuantityTensor)
    typing.assert_type(qt * t, _QuantityTensor)
    typing.assert_type(t * qt, _QuantityTensor)
    typing.assert_type(qt / t, _QuantityTensor)
    typing.assert_type(t / qt, _QuantityTensor)
    typing.assert_type(qt**2, _QuantityTensor)
    typing.assert_type(qt**0.5, _QuantityTensor)


def _mixed_scalar_and_tensor_quantities_give_a_tensor(q: _Quantity, qt: _QuantityTensor, t: _Tensor) -> None:
    typing.assert_type(q + qt, _QuantityTensor)
    typing.assert_type(qt + q, _QuantityTensor)
    typing.assert_type(q - qt, _QuantityTensor)
    typing.assert_type(qt - q, _QuantityTensor)
    typing.assert_type(q * qt, _QuantityTensor)
    typing.assert_type(qt * q, _QuantityTensor)
    typing.assert_type(q / qt, _QuantityTensor)
    typing.assert_type(qt / q, _QuantityTensor)
    typing.assert_type(q * t, _QuantityTensor)
    typing.assert_type(t * q, _QuantityTensor)
    typing.assert_type(t / q, _QuantityTensor)
    typing.assert_type(q / t, _QuantityTensor)


def _unitless_tensor_arithmetic_stays_unitless(t: _Tensor) -> None:
    typing.assert_type(t + t, _Tensor)
    typing.assert_type(t * 2.0, _Tensor)
    typing.assert_type(2.0 / t, _Tensor)
    typing.assert_type(-t, _Tensor)
    typing.assert_type(t**2, _Tensor)
    typing.assert_type(2.0**t, _Tensor)
    typing.assert_type(t * meter, _QuantityTensor)


def _reductions_and_selections_keep_the_carrier(qt: _QuantityTensor, t: _Tensor) -> None:
    typing.assert_type(qt.sum(time), _QuantityTensor)
    typing.assert_type(qt.mean(time), _QuantityTensor)
    typing.assert_type(qt.min(time), _QuantityTensor)
    typing.assert_type(qt.max(time), _QuantityTensor)
    typing.assert_type(qt.prod(time), _QuantityTensor)
    typing.assert_type(qt.isel(time, 0), _QuantityTensor)
    typing.assert_type(t.sum(time), _Tensor)
    typing.assert_type(qt.value, _Tensor)


def _union_arithmetic_type_checks(u: _Quantity | _QuantityTensor) -> None:
    typing.assert_type(u + u, _Quantity | _QuantityTensor)
    typing.assert_type(u - u, _Quantity | _QuantityTensor)
    typing.assert_type(u * u, _Quantity | _QuantityTensor)
    typing.assert_type(u / u, _Quantity | _QuantityTensor)
    typing.assert_type(u * 2.0, _Quantity | _QuantityTensor)
    typing.assert_type(u**2, _Quantity | _QuantityTensor)


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
