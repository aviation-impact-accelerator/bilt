"""The common quantity contract does not require shared implementation inheritance."""

from dataclasses import dataclass
from typing import assert_type

from bilt import Tensor, dim, tensor
from bilt.units import Quantity, Unit, kilogram
from bilt.units._quantity_protocol import QuantityLike


@dataclass(frozen=True)
class TensorQuantityExample:
    """Standalone representative of the future unit-bearing tensor type."""

    value: Tensor
    unit: Unit


def _unit_of(quantity: QuantityLike) -> Unit:
    return quantity.unit


def _scalar_value(quantity: QuantityLike[float]) -> float:
    return quantity.value


def _tensor_value(quantity: QuantityLike[Tensor]) -> Tensor:
    return quantity.value


def test_native_scalar_quantity_satisfies_protocol() -> None:
    scalar = Quantity(2.0, kilogram)
    assert_type(scalar.value, float)
    assert _scalar_value(scalar) == scalar.value
    assert _unit_of(scalar) is kilogram


def test_independent_tensor_quantity_satisfies_same_protocol() -> None:
    magnitude = tensor([1.0, 2.0], over=dim("time"))
    quantity = TensorQuantityExample(magnitude, kilogram)
    assert _unit_of(quantity) is kilogram
    assert _tensor_value(quantity) is magnitude


def test_protocol_default_accepts_either_magnitude() -> None:
    scalar = Quantity(2.0, kilogram)
    varied = TensorQuantityExample(tensor([1.0, 2.0], over=dim("time")), kilogram)
    for quantity in (scalar, varied):
        assert _unit_of(quantity) is kilogram
