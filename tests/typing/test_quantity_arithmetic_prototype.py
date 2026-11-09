"""Probe the limits of an arithmetic-capable quantity protocol.

This is an experiment, not a proposed public API. No production annotations
or runtime behaviour change. The negative cases are in the paired fixture.
"""

import typing

from bilt import Quantity, QuantityLike, QuantityTensor, Tensor
from bilt.units import kilogram


class ScalarArithmetic(QuantityLike[float], typing.Protocol):
    """The narrow signatures already implemented by concrete Quantity."""

    def __add__(self, other: Quantity) -> Quantity: ...

    def __mul__(self, other: Quantity | float) -> Quantity: ...


class ScalarEquationPorts(typing.Protocol):
    """A separate *scalar* view can type-check the nested equation."""

    hot: ScalarArithmetic
    heat: Quantity


def typed_scalar_supply(source: Quantity, resistance: Quantity, inner: ScalarEquationPorts) -> bool:
    """Works statically only when the inner ports are known to be scalar."""
    return source == inner.hot + resistance * inner.heat


def _static_tensor_contract(tensor: QuantityTensor) -> None:
    boundary: QuantityLike[Tensor] = tensor
    typing.assert_type(boundary.value, Tensor)


def test_concrete_scalar_satisfies_narrow_arithmetic_contract() -> None:
    source = Quantity(2.0, kilogram)
    scalar: ScalarArithmetic = source
    assert scalar.unit == kilogram

    if typing.TYPE_CHECKING:
        typing.assert_type(scalar + source, Quantity)
        typing.assert_type(scalar * source, Quantity)
