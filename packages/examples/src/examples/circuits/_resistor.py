"""Ideal linear resistor."""

from typing import Annotated

from bilt import Implementation, Interface, Quantity, QuantityTensor, equation, field, system
from bilt.units.dimensions import CURRENT, RESISTANCE, VOLTAGE
from examples.circuits._two_terminal import potential_difference


@system
class Resistor(metaclass=Interface):
    """Ideal linear resistor, following the two-terminal sign convention."""

    positive_potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
    negative_potential: Annotated[Quantity | QuantityTensor, VOLTAGE] = field()
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


resistor = Implementation(Resistor, [potential_difference, ohms_law])
