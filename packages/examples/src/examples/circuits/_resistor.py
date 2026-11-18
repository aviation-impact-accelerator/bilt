"""Ideal linear resistor with two pins."""

from typing import Annotated

from bilt_proposals import field

from bilt import Implementation, Interface, Quantity, QuantityTensor, equation, system
from bilt.units.dimensions import CURRENT, RESISTANCE, VOLTAGE
from examples.circuits._pin import Pin
from examples.circuits._two_terminal import (
    current_enters_at_positive_pin,
    current_is_conserved,
    voltage_is_potential_difference,
)


@system
class Resistor(metaclass=Interface):
    """Ideal linear resistor, following the two-terminal sign convention."""

    positive: Pin = field()
    negative: Pin = field()
    voltage: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    current: Annotated[Quantity | QuantityTensor, CURRENT] = field()
    resistance: Annotated[Quantity | QuantityTensor, RESISTANCE] = field()


@equation
def ohms_law(
    voltage: Annotated[Quantity, VOLTAGE],
    current: Annotated[Quantity, CURRENT],
    resistance: Annotated[Quantity, RESISTANCE],
) -> bool:
    """Ohm's law."""
    return voltage == current * resistance


ideal_resistor = Implementation(
    Resistor, [voltage_is_potential_difference, current_is_conserved, current_enters_at_positive_pin, ohms_law]
)
