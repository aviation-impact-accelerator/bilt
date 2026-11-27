"""Check the four public annotations: what satisfies each, and what arithmetic on them gives.

|          | Any shape  | Must be a tensor |
|----------|------------|------------------|
| Units    | `Quantity` | `QuantityTensor` |
| No units | `Number`   | `Tensor`         |

The typing-only functions are checked by pyrefly and ty over this module. Values a
protocol must reject are in `negative_quantity_cases.py.txt`. The runtime tests check
that `isinstance` agrees with the grid for the values users create.
"""

import typing

from bilt import Number, Quantity, QuantityTensor, Tensor, dim, tensor
from bilt.units import kilogram, meter

if typing.TYPE_CHECKING:
    from bilt.core._tensors._tensors import _QuantityTensor, _Tensor
    from bilt.units._quantities import _Quantity

time = dim("time")


def _concrete_values_satisfy_their_protocols(q: _Quantity, qt: _QuantityTensor, t: _Tensor) -> None:
    _: Quantity = q
    _ = qt
    _qt: QuantityTensor = qt
    _t: Tensor = t
    _n: Number = t
    _n = 2.0
    _n = 2


def _a_quantity_tensor_is_a_quantity(qt: QuantityTensor) -> Quantity:
    return qt


def _creating_values_gives_the_public_types() -> None:
    typing.assert_type(2.0 * kilogram, Quantity)
    typing.assert_type(tensor([1.0], over=time, unit=kilogram), QuantityTensor)
    typing.assert_type(tensor([1.0], over=time), Tensor)
    typing.assert_type(tensor([1.0], over=time) * meter, QuantityTensor)


def _quantity_arithmetic_stays_a_quantity(x: Quantity, y: Quantity) -> None:
    typing.assert_type(x + y, Quantity)
    typing.assert_type(x - y, Quantity)
    typing.assert_type(x * y, Quantity)
    typing.assert_type(x / y, Quantity)
    typing.assert_type(x * 2.0, Quantity)
    typing.assert_type(2.0 * x, Quantity)
    typing.assert_type(2.0 / x, Quantity)
    typing.assert_type(x**2, Quantity)
    typing.assert_type(-x, Quantity)
    typing.assert_type(x * meter, Quantity)
    typing.assert_type(x / meter, Quantity)
    typing.assert_type(x.convert_to(meter), Quantity)
    typing.assert_type(x.strip_metadata(), Quantity)
    typing.assert_type(x < y, bool)
    typing.assert_type(x.value, Number)


def _quantity_tensor_arithmetic_stays_a_quantity_tensor(x: QuantityTensor, q: Quantity, t: Tensor) -> None:
    typing.assert_type(x + x, QuantityTensor)
    typing.assert_type(x + q, QuantityTensor)
    typing.assert_type(x - q, QuantityTensor)
    typing.assert_type(x * q, QuantityTensor)
    typing.assert_type(x / q, QuantityTensor)
    typing.assert_type(x * t, QuantityTensor)
    typing.assert_type(t * x, QuantityTensor)
    typing.assert_type(x * 2.0, QuantityTensor)
    typing.assert_type(2.0 * x, QuantityTensor)
    typing.assert_type(x**2, QuantityTensor)
    typing.assert_type(x * meter, QuantityTensor)
    typing.assert_type(x.sum(time), QuantityTensor)
    typing.assert_type(x.isel(time, 0), QuantityTensor)
    typing.assert_type(x.value, Tensor)


def _a_unitless_tensor_gains_a_unit_only_from_a_quantity_or_unit(t: Tensor, q: Quantity) -> None:
    typing.assert_type(t + t, Tensor)
    typing.assert_type(t + 1.0, Tensor)
    typing.assert_type(1.0 - t, Tensor)
    typing.assert_type(t * 2.0, Tensor)
    typing.assert_type(t**2, Tensor)
    typing.assert_type(t.mean(time), Tensor)
    typing.assert_type(t * q, QuantityTensor)
    typing.assert_type(q * t, QuantityTensor)
    typing.assert_type(t / q, QuantityTensor)
    typing.assert_type(t * meter, QuantityTensor)


def _number_arithmetic_stays_a_number(n: Number) -> None:
    typing.assert_type(n + n, Number)
    typing.assert_type(n * 2.0, Number)
    typing.assert_type(n + 1.0, Number)
    typing.assert_type(n**2, Number)


def test_isinstance_follows_the_grid_for_created_values() -> None:
    scalar = 2.0 * kilogram
    labelled = tensor([1.0, 2.0], over=time, unit=kilogram)
    unitless = tensor([1.0, 2.0], over=time)

    assert isinstance(scalar, Quantity)
    assert not isinstance(scalar, Tensor)
    assert not isinstance(scalar, QuantityTensor)

    assert isinstance(labelled, Quantity)
    assert isinstance(labelled, QuantityTensor)

    assert isinstance(unitless, Tensor)
    assert not isinstance(unitless, Quantity)
    assert isinstance(unitless, Number)
    assert isinstance(2.0, Number)
