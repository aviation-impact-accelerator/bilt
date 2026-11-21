"""Ideal voltage source with two pins."""

from typing import Annotated

from bilt_proposals import field

from bilt import Implementation, Interface, Quantity, QuantityTensor, system
from bilt.units.dimensions import CURRENT, VOLTAGE
from examples.circuits._pin import Pin
from examples.circuits._two_terminal import (
    current_enters_at_positive_pin,
    current_is_conserved,
    voltage_is_potential_difference,
)


@system
class VoltageSource(metaclass=Interface):
    """Ideal source: `voltage` is the EMF, the positive pin at the higher potential."""

    positive: Pin = field()
    negative: Pin = field()
    voltage: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    current: Annotated[Quantity | QuantityTensor, CURRENT] = field()


ideal_voltage_source = Implementation(
    VoltageSource, [voltage_is_potential_difference, current_is_conserved, current_enters_at_positive_pin]
)
