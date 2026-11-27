"""Static contracts for the proposed internal expression protocols.

These functions are checked by ty and Pyrefly, not executed as eager arithmetic.
Concrete implementation conformance is part of the subsequent API transition.
"""

import typing

from bilt import dim

if typing.TYPE_CHECKING:
    from bilt.core._tensors._protocols import Number, QuantityTensor, Tensor
    from bilt.units._protocols import Quantity


time = dim("time")


def _quantity_arithmetic(q: Quantity, other: Quantity, qt: QuantityTensor, t: Tensor) -> None:
    typing.assert_type(q + other, Quantity)
    typing.assert_type(q * 2.0, Quantity)
    typing.assert_type(q + qt, QuantityTensor)
    typing.assert_type(q * t, QuantityTensor)
    typing.assert_type(q.value, Number)


def _tensor_arithmetic(qt: QuantityTensor, q: Quantity, t: Tensor) -> None:
    typing.assert_type(qt + q, QuantityTensor)
    typing.assert_type(qt * t, QuantityTensor)
    typing.assert_type(qt.sum(time), QuantityTensor)
    typing.assert_type(qt.isel(time, 0), QuantityTensor)
    typing.assert_type(qt.value, Tensor)
    typing.assert_type(t + 2.0, Tensor)
    typing.assert_type(t * q, QuantityTensor)
    typing.assert_type(t.mean(time), Tensor)


def _quantity_tensor_is_quantity(qt: QuantityTensor) -> Quantity:
    return qt
