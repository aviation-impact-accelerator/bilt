"""Ideal voltage source with two pins."""

from typing import Annotated

from bilt_proposals import field

from bilt import Implementation, Interface, Quantity, QuantityTensor, system
from bilt.units.dimensions import CURRENT, VOLTAGE
from examples.circuits._pin import Pin
from examples.circuits._two_terminal import current_conservation, current_through, potential_difference


@system
class VoltageSource(metaclass=Interface):
    """Ideal source: `voltage` is the EMF, the positive pin at the higher potential."""

    positive: Pin = field()
    negative: Pin = field()
    voltage: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    current: Annotated[Quantity | QuantityTensor, CURRENT] = field()


voltage_source = Implementation(VoltageSource, [potential_difference, current_conservation, current_through])
