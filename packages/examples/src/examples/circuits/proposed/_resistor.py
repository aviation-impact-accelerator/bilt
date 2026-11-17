"""Ideal linear resistor with two pins."""

from typing import Annotated

from bilt_proposals import field

from bilt import Implementation, Interface, Quantity, QuantityTensor, system
from bilt.units.dimensions import CURRENT, RESISTANCE, VOLTAGE
from examples.circuits._resistor import ohms_law
from examples.circuits.proposed._pin import Pin
from examples.circuits.proposed._two_terminal import current_conservation, current_through, potential_difference


@system
class Resistor(metaclass=Interface):
    """Ideal linear resistor, following the two-terminal sign convention."""

    positive: Pin = field()
    negative: Pin = field()
    voltage: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    current: Annotated[Quantity | QuantityTensor, CURRENT] = field()
    resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()


# `ohms_law` is reused unchanged from the flat resistor: equations bind by port name.
resistor = Implementation(Resistor, [potential_difference, current_conservation, current_through, ohms_law])
